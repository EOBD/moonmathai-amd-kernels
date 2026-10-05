// Torch bindings for the gfx950 (CDNA4, MI355X) bf16 attention kernels: moonmath_amd._C_gfx950.
//
//   forward         one batch, layout "bshd" or "bhsd": q [B, Sq, H, Dqk], k [B, Skv, HKv, Dqk],
//                   v [B, Skv, HKv, Dv] -> o [B, Sq, H, Dv] (bshd; bhsd swaps S and H), lse [B, H, Sq]
//   forward_varlen  THD: q [Tq, H, Dqk], k [Tk, HKv, Dqk], v [Tk, HKv, Dv] -> o [Tq, H, Dv], lse [H, Tq];
//                   sequence i spans q tokens cu_seqlens_q[i] .. cu_seqlens_q[i + 1] - 1, k/v likewise
//   VarlenPlan      forward_varlen planned once (per forward pass) and reused on each layer's tensors
//   forward_paged   forward_varlen with K/V in pages: k_cache [pages, page_size, HKv, Dqk], v_cache
//                   [pages, page_size, HKv, Dv]; a page table (aiter mha_batch_prefill, vLLM blocks)
//
// Any strides are allowed (views, slices, padding, per-tensor layouts) as long as the head dim is
// contiguous: the kernels index each tensor by its own batch, token and head strides, and a new output is
// contiguous in the call's layout. (Dqk, Dv) is (128, 128) or (192, 128) (MLA prefill); GQA when
// H % HKv == 0; causal is bottom-right aligned. lse (optional) is each q row's fp32 natural-log
// log-sum-exp in AITER's layout, -inf for a row that sees no key. Everything runs on the current torch
// stream and refuses any device but gfx950. The kernels' C API lives in attention.h (head dim 128
// entries take D = 128; the MLA entries none).

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime_api.h>
#include <torch/extension.h>

#include <array>
#include <climits>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "attention.h"

namespace py = pybind11;

namespace {

using Result = std::tuple<at::Tensor, std::optional<at::Tensor>>;  // out, plus lse when requested

[[noreturn]] void fail(const std::string& who, const std::string& what) {
  throw std::invalid_argument(who + ": " + what);
}

std::string shape_str(const at::Tensor& t) {
  std::ostringstream s;
  s << t.sizes();
  return s.str();
}

std::string strides_str(const at::Tensor& t) {
  std::ostringstream s;
  s << t.strides();
  return s.str();
}

// ---- tensors and layout -------------------------------------------------------------------------

// Which dims hold a tensor's batch, token and head (batch -1: none, THD).
struct Dims {
  int b, s, h;
};
constexpr Dims THD{-1, 0, 1};
constexpr Dims Pages{0, 1, 2};       // shape [pages, page_size, heads, d]
constexpr Dims BatchLse{0, 2, 1};    // shape [B, H, Sq]
constexpr Dims VarlenLse{-1, 1, 0};  // shape [H, Tq]

Dims layout_dims(const std::string& who, const std::string& layout) {
  if (layout == "bshd") return {0, 1, 2};
  if (layout == "bhsd") return {0, 2, 1};
  fail(who, "layout must be 'bshd' or 'bhsd', got '" + layout + "'");
}

// bf16, `dim`-D, contiguous head dim, resident on q's GPU.
void check_tensor(const std::string& who, const at::Tensor& t, const char* name, int64_t dim, const at::Tensor& q) {
  const std::string n = name;
  if (t.scalar_type() != at::kBFloat16)
    fail(who, n + " must be bfloat16, got " + std::string(at::toString(t.scalar_type())));
  if (t.dim() != dim) fail(who, n + " must be " + std::to_string(dim) + "-D, got shape " + shape_str(t));
  if (!t.device().is_cuda()) fail(who, n + " must be on a GPU, got " + t.device().str());
  if (t.device() != q.device())
    fail(who, n + " must be on q's device (" + q.device().str() + "), got " + t.device().str());
  if (t.stride(-1) != 1)
    fail(who, n + " must have a contiguous last dim (stride(-1) == 1), got stride " + strides_str(t));
}

// The kernels target gfx950 only: reject any other device, like AITER's get_gpu_arch() guard.
void require_gfx950(const std::string& who, const at::Device& device) {
  std::string arch = at::cuda::getDeviceProperties(device.index())->gcnArchName;
  arch = arch.substr(0, arch.find(':'));
  if (arch != "gfx950")
    throw std::runtime_error(
        who + ": moonmath_amd._C_gfx950 runs on gfx950 (CDNA4) only; " + device.str() + " is " + arch);
}

// Whether (Dqk, Dv) are MLA's (192, 128); otherwise they must be (128, 128).
bool is_mla(const std::string& who, int64_t Dqk, int64_t Dv) {
  const bool mla = Dqk == 192 && Dv == 128;
  if (!mla && !(Dqk == 128 && Dv == 128))
    fail(
        who,
        "head dims (q/k, v) must be (128, 128) or (192, 128), got (" + std::to_string(Dqk) + ", " + std::to_string(Dv) +
            ")");
  return mla;
}

// Require H to be a multiple of HKv (MHA, GQA, MQA).
void check_heads(const std::string& who, int64_t H, int64_t HKv, const char* q_heads, const char* kv_heads) {
  if (HKv < 1 || H % HKv != 0)
    fail(
        who,
        std::string(q_heads) + " (" + std::to_string(H) + ") must be a multiple of " + kv_heads + " (" +
            std::to_string(HKv) + ")");
}

// q, k, v: bf16 `dim`-D tensors on a single gfx950 GPU, heads in dim h, valid head dims. Returns whether it is MLA.
bool check_qkv(const std::string& who, const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, int dim, int h) {
  check_tensor(who, q, "q", dim, q);
  require_gfx950(who, q.device());
  check_tensor(who, k, "k", dim, q);
  check_tensor(who, v, "v", dim, q);
  const int64_t Dqk = q.size(-1);
  const bool mla = is_mla(who, Dqk, v.size(-1));
  if (k.size(-1) != Dqk) fail(who, "k's head dim must match q's (" + std::to_string(Dqk) + "), got k " + shape_str(k));
  if (v.size(h) != k.size(h))
    fail(who, "k and v must have the same heads, got k " + shape_str(k) + ", v " + shape_str(v));
  check_heads(who, q.size(h), k.size(h), "q heads", "k/v heads");
  return mla;
}

float check_scale(const std::string& who, std::optional<double> softmax_scale, int64_t Dqk) {
  const double s = softmax_scale.value_or(1.0 / std::sqrt((double)Dqk));
  if (!(s > 0 && s < 3.4e38)) fail(who, "softmax_scale must be positive and finite, got " + std::to_string(s));
  return (float)s;
}

int to_int(const std::string& who, const char* name, int64_t x) {
  if (x < 0 || x > INT_MAX) fail(who, std::string(name) + " must be in [0, 2^31 - 1], got " + std::to_string(x));
  return (int)x;
}

// A call's outputs: out (supplied or fresh, bf16) and lse (fresh fp32, when requested).
struct Outputs {
  at::Tensor out;
  std::optional<at::Tensor> lse;
};

Outputs outputs(
    const std::string& who,
    const std::optional<at::Tensor>& out,
    at::IntArrayRef out_sizes,
    bool return_lse,
    at::IntArrayRef lse_sizes,
    const at::Tensor& q) {
  Outputs o;
  if (out) {
    check_tensor(who, *out, "out", (int64_t)out_sizes.size(), q);
    if (out->sizes() != out_sizes) {
      std::ostringstream s;
      s << "out must have shape " << out_sizes << ", got " << out->sizes();
      fail(who, s.str());
    }
    o.out = *out;
  } else {
    o.out = at::empty(out_sizes, q.options());
  }
  if (return_lse) o.lse = at::empty(lse_sizes, q.options().dtype(at::kFloat));
  return o;
}

// Tokens lie at most 2^22 elements apart: the kernels' token strides are 32-bit (attention_plan.hpp MaxToken).
constexpr int64_t MaxTokenStride = int64_t(1) << 22;

// The C entries' leading arguments: tensor data pointers and their strides (attention.h).
struct Tensors {
  const void *q, *k, *v;
  void* o;
  float* lse;
  std::array<int64_t, 15> strides;  // batch, token and head strides for q, k, v, out, then lse's
};

// q, k, v and out with strides: q and out use dims d, k and v use kv_d, lse uses lse_d.
Tensors tensors(
    const std::string& who,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const Outputs& o,
    Dims d,
    Dims kv_d,
    Dims lse_d) {
  Tensors t{q.data_ptr(), k.data_ptr(), v.data_ptr(), o.out.data_ptr(), o.lse ? o.lse->data_ptr<float>() : nullptr};
  const at::Tensor* ts[5] = {&q, &k, &v, &o.out, o.lse ? &*o.lse : nullptr};
  const char* names[5] = {"q", "k", "v", "out", "lse"};
  for (int i = 0; i < 5 && ts[i]; ++i) {
    const Dims di = i == 4 ? lse_d : i == 1 || i == 2 ? kv_d : d;
    int64_t* st = &t.strides[3 * i];
    st[0] = di.b >= 0 ? ts[i]->stride(di.b) : 0;
    st[1] = ts[i]->stride(di.s);
    st[2] = ts[i]->stride(di.h);
    if (st[1] > MaxTokenStride)
      fail(
          who,
          std::string(names[i]) + "'s tokens must be at most 2^22 elements apart, got stride " + strides_str(*ts[i]));
  }
  return t;
}

// 0: queued; -1: the kernels rejected the arguments; otherwise a hipError_t.
void check_rc(const std::string& who, int rc) {
  if (rc == -1) fail(who, "the kernel rejected its arguments");
  if (rc != 0) throw std::runtime_error(who + ": launch failed: " + hipGetErrorString((hipError_t)rc));
}

hipStream_t current_stream(const at::Device& device) {
  return at::cuda::getCurrentCUDAStream(device.index()).stream();
}

// ---- the cu_seqlens path -------------------------------------------------------------------------

// A varlen call's sequences: cu_seqlens on device, on host, or both (host copies of the device ones).
struct Seqs {
  const int *dev_q = nullptr, *dev_k = nullptr;    // null: no device tensor
  const int *host_q = nullptr, *host_k = nullptr;  // null: no host tensor
  int n = 0;                                       // sequence count
};

// Contiguous int32 [num_seqs + 1], on host memory or on `device`.
void check_cu_tensor(const std::string& who, const at::Tensor& cu, const char* name, const at::Device& device) {
  if (cu.scalar_type() != at::kInt || cu.dim() != 1 || !cu.is_contiguous() || cu.numel() < 1)
    fail(who, std::string(name) + " must be a contiguous 1-D int32 tensor of num_seqs + 1 entries");
  if (!cu.device().is_cpu() && cu.device() != device)
    fail(who, std::string(name) + " must be in host memory or on " + device.str() + ", got " + cu.device().str());
}

// Host values must be non-decreasing, within [0, tokens], and no longer than max_len per sequence
// (max_seqlen sizes a device plan).
void check_cu_values(const std::string& who, const at::Tensor& cu, const char* name, int64_t tokens, int max_len) {
  const std::string n = name;
  const int* c = cu.data_ptr<int>();
  if (c[0] < 0) fail(who, n + "[0] must be >= 0");
  for (int64_t i = 0; i + 1 < cu.numel(); ++i) {
    if (c[i + 1] < c[i]) fail(who, n + " must be non-decreasing");
    if (c[i + 1] - c[i] > max_len)
      fail(
          who,
          n + " has a sequence of " + std::to_string(c[i + 1] - c[i]) + " tokens, more than max_seqlen " +
              std::to_string(max_len));
  }
  const int end = c[cu.numel() - 1];
  if (end > tokens)
    fail(who, n + " ends at " + std::to_string(end) + ", past the " + std::to_string(tokens) + " tokens");
}

// cu_seqlens_q / _k on the host, or on `device` with optional host copies. Host values get checked;
// device ones cannot be without a sync.
Seqs check_seqs(
    const std::string& who,
    const at::Tensor& cu_q,
    const at::Tensor& cu_k,
    const std::optional<at::Tensor>& host_cu_q,
    const std::optional<at::Tensor>& host_cu_k,
    int max_q,
    int max_k,
    int64_t total_q,
    int64_t total_k,
    const at::Device& device) {
  check_cu_tensor(who, cu_q, "cu_seqlens_q", device);
  check_cu_tensor(who, cu_k, "cu_seqlens_k", device);
  if (cu_q.numel() != cu_k.numel()) fail(who, "cu_seqlens_q and cu_seqlens_k must have the same length");
  if (cu_q.device() != cu_k.device()) fail(who, "cu_seqlens_q and cu_seqlens_k must be on the same device");
  if (host_cu_q.has_value() != host_cu_k.has_value())
    fail(who, "give host_cu_seqlens_q and host_cu_seqlens_k both or neither");
  Seqs s;
  s.n = (int)cu_q.numel() - 1;
  if (cu_q.device().is_cpu()) {
    if (host_cu_q) fail(who, "host_cu_seqlens_q / _k are host copies of GPU cu_seqlens; these are on the host");
    check_cu_values(who, cu_q, "cu_seqlens_q", total_q, max_q);
    check_cu_values(who, cu_k, "cu_seqlens_k", total_k, max_k);
    s.host_q = cu_q.data_ptr<int>(), s.host_k = cu_k.data_ptr<int>();
    return s;
  }
  s.dev_q = cu_q.data_ptr<int>(), s.dev_k = cu_k.data_ptr<int>();
  if (host_cu_q) {
    for (const auto* t : {&*host_cu_q, &*host_cu_k})
      if (!t->device().is_cpu() || t->scalar_type() != at::kInt || !t->is_contiguous() || t->numel() != cu_q.numel())
        fail(who, "host_cu_seqlens_q / _k must be contiguous int32 CPU tensors of cu_seqlens_q's length");
    check_cu_values(who, *host_cu_q, "host_cu_seqlens_q", total_q, max_q);
    check_cu_values(who, *host_cu_k, "host_cu_seqlens_k", total_k, max_k);
    s.host_q = host_cu_q->data_ptr<int>(), s.host_k = host_cu_k->data_ptr<int>();
  }
  return s;
}

// ---- forward and forward_varlen ------------------------------------------------------------------

// A batch of B sequences, in BSHD or BHSD.
Result forward(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    std::optional<at::Tensor> out,
    bool causal,
    std::optional<double> softmax_scale,
    const std::string& layout,
    bool return_lse) {
  const std::string who = "forward (gfx950)";
  const Dims d = layout_dims(who, layout);
  const bool mla = check_qkv(who, q, k, v, 4, d.h);
  const int64_t B = q.size(0), Sq = q.size(d.s), H = q.size(d.h), Skv = k.size(d.s), HKv = k.size(d.h), Dv = v.size(3);
  if (k.size(0) != B || v.size(0) != B || v.size(d.s) != Skv)
    fail(
        who,
        "q, k, v must be " + std::string(layout == "bshd" ? "[B, S, H, D]" : "[B, H, S, D]") +
            " with k and v alike, got q " + shape_str(q) + ", k " + shape_str(k) + ", v " + shape_str(v));
  if (B * Sq > INT_MAX || B * Skv > INT_MAX) fail(who, "at most 2^31 - 1 tokens");
  const float scale = check_scale(who, softmax_scale, q.size(3));
  const auto out_sizes = layout == "bshd" ? std::vector<int64_t>{B, Sq, H, Dv} : std::vector<int64_t>{B, H, Sq, Dv};
  const Outputs o = outputs(who, out, out_sizes, return_lse, {B, H, Sq}, q);
  if (o.out.numel() == 0) return {o.out, o.lse};
  const Tensors t = tensors(who, q, k, v, o, d, d, BatchLse);

  const c10::cuda::CUDAGuard device_guard(q.device());
  const hipStream_t stream = current_stream(q.device());
  const int64_t* st = t.strides.data();
  check_rc(
      who,
      mla ? cdna4_attn_mla_fwd_bf16_strided(t.q, t.k, t.v, t.o, t.lse, st, B, H, HKv, Sq, Skv, causal, scale, stream)
          : cdna4_attn_fwd_bf16_strided(t.q, t.k, t.v, t.o, t.lse, st, B, H, HKv, Sq, Skv, 128, causal, scale, stream));
  return {o.out, o.lse};
}

// Variable-length sequences, THD (any token and head strides). Host cu_seqlens take the host planner;
// device cu_seqlens take the device path (planned on GPU, or on host from host copies when not capturing).
Result forward_varlen(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k,
    int64_t max_seqlen_q,
    int64_t max_seqlen_k,
    std::optional<at::Tensor> out,
    bool causal,
    std::optional<double> softmax_scale,
    bool return_lse,
    std::optional<at::Tensor> host_cu_seqlens_q,
    std::optional<at::Tensor> host_cu_seqlens_k) {
  const std::string who = "forward_varlen (gfx950)";
  const bool mla = check_qkv(who, q, k, v, 3, 1);
  const int64_t Tq = q.size(0), H = q.size(1), Tk = k.size(0), HKv = k.size(1), Dv = v.size(2);
  if (v.size(0) != Tk) fail(who, "k and v must have the same tokens, got k " + shape_str(k) + ", v " + shape_str(v));
  const int tq = to_int(who, "q's tokens", Tq), tk = to_int(who, "k's tokens", Tk);
  const int mq = to_int(who, "max_seqlen_q", max_seqlen_q), mk = to_int(who, "max_seqlen_k", max_seqlen_k);
  const Seqs s =
      check_seqs(who, cu_seqlens_q, cu_seqlens_k, host_cu_seqlens_q, host_cu_seqlens_k, mq, mk, Tq, Tk, q.device());
  const float scale = check_scale(who, softmax_scale, q.size(2));
  const Outputs o = outputs(who, out, {Tq, H, Dv}, return_lse, {H, Tq}, q);
  if (o.out.numel() == 0) return {o.out, o.lse};
  const Tensors t = tensors(who, q, k, v, o, THD, THD, VarlenLse);

  const c10::cuda::CUDAGuard device_guard(q.device());
  const hipStream_t stream = current_stream(q.device());
  const int64_t* st = t.strides.data();
  const int *dq = s.dev_q, *dk = s.dev_k, *hq = s.host_q, *hk = s.host_k;
  int rc;
  if (!dq)  // host cu_seqlens path
    rc = mla ? cdna4_attn_mla_varlen_fwd_bf16_strided(
                   t.q, t.k, t.v, t.o, t.lse, st, hq, hk, s.n, H, HKv, causal, scale, stream)
             : cdna4_attn_varlen_fwd_bf16_strided(
                   t.q, t.k, t.v, t.o, t.lse, st, hq, hk, s.n, H, HKv, 128, causal, scale, stream);
  else if (mla)
    rc = cdna4_attn_mla_varlen_fwd_bf16_dev_strided(
        t.q, t.k, t.v, t.o, t.lse, st, dq, dk, hq, hk, s.n, mq, mk, tq, tk, H, HKv, causal, scale, stream);
  else
    rc = cdna4_attn_varlen_fwd_bf16_dev_strided(
        t.q, t.k, t.v, t.o, t.lse, st, dq, dk, hq, hk, s.n, mq, mk, tq, tk, H, HKv, 128, causal, scale, stream);
  check_rc(who, rc);
  return {o.out, o.lse};
}

// ---- the forward_paged path ----------------------------------------------------------------------

// A contiguous 1-D int32 host tensor of n entries (n < 0: any length).
const int* host_ints(const std::string& who, const at::Tensor& t, const char* name, int64_t n) {
  if (t.scalar_type() != at::kInt || t.dim() != 1 || !t.is_contiguous() || !t.device().is_cpu() ||
      (n >= 0 && t.numel() != n))
    fail(
        who,
        std::string(name) + " must be a contiguous 1-D int32 CPU tensor" +
            (n >= 0 ? " of " + std::to_string(n) + " entries" : ""));
  return t.data_ptr<int>();
}

// Variable-length sequences with paged K/V: q THD, k_cache / v_cache [pages, page_size, HKv, D] (any
// strides). Sequence i has q tokens cu_seqlens_q[i] .. cu_seqlens_q[i + 1] - 1 and seqlens_k[i] keys; key r
// is token r % page_size within page page_table[page_starts[i] + r / page_size]. cu_seqlens_q, page_starts
// and seqlens_k are on the host, page_table on the GPU (unchecked: it must hold the caches' pages).
Result forward_paged(
    const at::Tensor& q,
    const at::Tensor& k_cache,
    const at::Tensor& v_cache,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& page_table,
    const at::Tensor& page_starts,
    const at::Tensor& seqlens_k,
    std::optional<at::Tensor> out,
    bool causal,
    std::optional<double> softmax_scale,
    bool return_lse) {
  const std::string who = "forward_paged (gfx950)";
  check_tensor(who, q, "q", 3, q);
  require_gfx950(who, q.device());
  check_tensor(who, k_cache, "k_cache", 4, q);
  check_tensor(who, v_cache, "v_cache", 4, q);
  const int64_t Tq = q.size(0), H = q.size(1), Dqk = q.size(2), HKv = k_cache.size(2), Dv = v_cache.size(3);
  const bool mla = is_mla(who, Dqk, Dv);
  if (k_cache.size(3) != Dqk || !k_cache.sizes().slice(0, 3).equals(v_cache.sizes().slice(0, 3)))
    fail(
        who,
        "k_cache and v_cache must be [pages, page_size, kv_heads, D] alike, with q's head dim, got k_cache " +
            shape_str(k_cache) + ", v_cache " + shape_str(v_cache));
  check_heads(who, H, HKv, "q heads", "kv heads");
  const int P = to_int(who, "page_size", k_cache.size(1)), min_page = mla ? 16 : 8;
  if (P < min_page || (P & (P - 1)))
    fail(who, "page_size must be a power of two, at least " + std::to_string(min_page) + ", got " + std::to_string(P));
  if (!mla && k_cache.strides() != v_cache.strides())
    fail(
        who,
        "k_cache and v_cache must have the same strides (head dim 128), got " + strides_str(k_cache) + ", " +
            strides_str(v_cache));
  if (page_table.scalar_type() != at::kInt || page_table.dim() != 1 || !page_table.is_contiguous() ||
      page_table.device() != q.device() || page_table.numel() < 1 || page_table.numel() > INT_MAX / 4)
    fail(who, "page_table must be a non-empty contiguous 1-D int32 tensor on " + q.device().str());
  const int n = to_int(who, "num_seqs", cu_seqlens_q.numel() - 1), table_len = (int)page_table.numel();
  const int* cu_q = host_ints(who, cu_seqlens_q, "cu_seqlens_q", -1);
  const int* starts = host_ints(who, page_starts, "page_starts", n);
  const int* lens = host_ints(who, seqlens_k, "seqlens_k", n);
  check_cu_values(who, cu_seqlens_q, "cu_seqlens_q", Tq, INT_MAX);
  for (int i = 0; i < n; ++i)
    if (starts[i] < 0 || lens[i] < 0 || (lens[i] + int64_t(P) - 1) / P > table_len - starts[i])
      fail(
          who,
          "sequence " + std::to_string(i) + "'s " + std::to_string(lens[i]) + " keys from page_starts " +
              std::to_string(starts[i]) + " run past the page table's " + std::to_string(table_len) + " entries");
  const float scale = check_scale(who, softmax_scale, Dqk);
  const Outputs o = outputs(who, out, {Tq, H, Dv}, return_lse, {H, Tq}, q);
  if (o.out.numel() == 0) return {o.out, o.lse};
  const Tensors t = tensors(who, q, k_cache, v_cache, o, THD, Pages, VarlenLse);

  const c10::cuda::CUDAGuard device_guard(q.device());
  const hipStream_t stream = current_stream(q.device());
  const int64_t* st = t.strides.data();
  const int* table = page_table.data_ptr<int>();
  int rc;
  if (mla)
    rc = cdna4_attn_mla_paged_fwd_bf16_strided(
        t.q, t.k, t.v, t.o, t.lse, st, cu_q, table, table_len, starts, lens, n, P, H, HKv, causal, scale, stream);
  else
    rc = cdna4_attn_paged_fwd_bf16_strided(
        t.q, t.k, t.v, t.o, t.lse, st, cu_q, table, table_len, starts, lens, n, P, H, HKv, 128, causal, scale, stream);
  check_rc(who, rc);
  return {o.out, o.lse};
}

// ---- the VarlenPlan class ------------------------------------------------------------------------

// Whether `device`'s current stream is capturing a graph.
bool capturing(int device) {
  hipStreamCaptureStatus s = hipStreamCaptureStatusNone;
  (void)hipStreamIsCapturing(at::cuda::getCurrentCUDAStream(device).stream(), &s);
  return s != hipStreamCaptureStatusNone;
}

// Closed handles wait here while their device is capturing a graph, since freeing device memory would
// break the capture. Each close destroys the waiting handles whose device has stopped capturing.
std::mutex closed_mutex;
std::vector<std::pair<void*, int>> closed_handles;  // {handle, device}

void destroy_when_safe(void* handle, int device) {
  std::lock_guard<std::mutex> lock(closed_mutex);
  closed_handles.emplace_back(handle, device);
  for (size_t i = 0; i < closed_handles.size();) {
    const auto [h, dev] = closed_handles[i];
    if (capturing(dev)) {
      ++i;
      continue;
    }
    const c10::cuda::CUDAGuard device_guard((c10::DeviceIndex)dev);
    cdna4_attn_plan_destroy(h);
    closed_handles[i] = closed_handles.back();
    closed_handles.pop_back();
  }
}

// forward_varlen planned once and run repeatedly, wrapping a C API plan handle. A graph that captured
// plan() or run() reads the handle's buffers, so they must live until close().
class VarlenPlan {
 public:
  VarlenPlan() = default;
  VarlenPlan(const VarlenPlan&) = delete;
  VarlenPlan& operator=(const VarlenPlan&) = delete;
  ~VarlenPlan() {
    close();
  }

  void plan(
      const at::Tensor& cu_seqlens_q,
      const at::Tensor& cu_seqlens_k,
      int64_t max_seqlen_q,
      int64_t max_seqlen_k,
      int64_t total_q,
      int64_t total_k,
      int64_t heads,
      int64_t kv_heads,
      std::tuple<int64_t, int64_t> head_dims,
      bool causal,
      std::optional<at::Tensor> host_cu_seqlens_q,
      std::optional<at::Tensor> host_cu_seqlens_k) {
    const std::string who = "VarlenPlan.plan (gfx950)";
    const at::Device device = cu_seqlens_q.device();
    if (!device.is_cuda()) fail(who, "cu_seqlens_q must be on the GPU");
    require_gfx950(who, device);
    if (handle_ && device.index() != device_)
      fail(who, "a plan stays on its first device, cuda:" + std::to_string(device_));
    const bool mla = is_mla(who, std::get<0>(head_dims), std::get<1>(head_dims));
    const int H = to_int(who, "heads", heads), HKv = to_int(who, "kv_heads", kv_heads);
    check_heads(who, H, HKv, "heads", "kv_heads");
    const int max_q = to_int(who, "max_seqlen_q", max_seqlen_q), max_k = to_int(who, "max_seqlen_k", max_seqlen_k);
    const int tq = to_int(who, "total_q", total_q), tk = to_int(who, "total_k", total_k);
    const Seqs s =
        check_seqs(who, cu_seqlens_q, cu_seqlens_k, host_cu_seqlens_q, host_cu_seqlens_k, max_q, max_k, tq, tk, device);

    const c10::cuda::CUDAGuard device_guard(device);
    if (!handle_ && !(handle_ = cdna4_attn_plan_create())) throw std::runtime_error(who + ": out of host memory");
    device_ = device.index();
    planned_ = false;
    const hipStream_t stream = current_stream(device);
    const int *dq = s.dev_q, *dk = s.dev_k, *hq = s.host_q, *hk = s.host_k;
    check_rc(
        who,
        mla ? cdna4_attn_mla_varlen_plan(handle_, dq, dk, hq, hk, s.n, max_q, max_k, tq, tk, H, HKv, causal, stream)
            : cdna4_attn_varlen_plan(handle_, dq, dk, hq, hk, s.n, max_q, max_k, tq, tk, H, HKv, 128, causal, stream));
    planned_ = true, mla_ = mla, causal_ = causal, H_ = H, HKv_ = HKv, total_q_ = tq, total_k_ = tk;
  }

  Result
  run(const at::Tensor& q,
      const at::Tensor& k,
      const at::Tensor& v,
      std::optional<at::Tensor> out,
      std::optional<double> softmax_scale,
      bool return_lse) const {
    const std::string who = "VarlenPlan.run (gfx950)";
    if (!planned_) throw std::runtime_error(who + ": plan() it first (after close(), too)");
    if (check_qkv(who, q, k, v, 3, 1) != mla_) fail(who, "q, k, v's head dims differ from the plan's");
    if (q.device().index() != device_) fail(who, "q must be on the plan's device, cuda:" + std::to_string(device_));
    if (q.size(0) != total_q_ || q.size(1) != H_ || k.size(0) != total_k_ || k.size(1) != HKv_ || v.size(0) != total_k_)
      fail(
          who,
          "q must be [" + std::to_string(total_q_) + ", " + std::to_string(H_) + ", D] and k, v [" +
              std::to_string(total_k_) + ", " + std::to_string(HKv_) + ", D] as planned, got q " + shape_str(q) +
              ", k " + shape_str(k) + ", v " + shape_str(v));
    const float scale = check_scale(who, softmax_scale, q.size(2));
    const Outputs o = outputs(who, out, {total_q_, H_, v.size(2)}, return_lse, {H_, total_q_}, q);
    if (o.out.numel() == 0) return {o.out, o.lse};
    const Tensors t = tensors(who, q, k, v, o, THD, THD, VarlenLse);

    const c10::cuda::CUDAGuard device_guard(q.device());
    const hipStream_t stream = current_stream(q.device());
    const int64_t* st = t.strides.data();
    const int rc = mla_ ? cdna4_attn_mla_varlen_run(handle_, t.q, t.k, t.v, t.o, t.lse, st, scale, stream)
                        : cdna4_attn_varlen_run(handle_, t.q, t.k, t.v, t.o, t.lse, st, scale, stream);
    if (rc == -1)  // arguments were checked, so the capture state differs
      fail(who, "plan and run must be in the same graph capture, or both outside one: re-plan it");
    check_rc(who, rc);
    return {o.out, o.lse};
  }

  // Frees the handle once its kernels finish (or later, if a capture is running). Re-planning creates a new one.
  void close() {
    if (handle_) destroy_when_safe(handle_, device_);
    handle_ = nullptr, planned_ = false;
  }

  bool planned() const {
    return planned_;
  }
  bool causal() const {
    return causal_;
  }

 private:
  void* handle_ = nullptr;
  int device_ = -1;
  bool planned_ = false, mla_ = false, causal_ = false;
  int64_t H_ = 0, HKv_ = 0, total_q_ = 0, total_k_ = 0;  // heads and tokens for each run
};

}  // anonymous namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "moonmath_amd: gfx950 (CDNA4, MI355X) bf16 attention kernels";

  m.def(
      "forward",
      &forward,
      "Attention over a batch, layout 'bshd' or 'bhsd', any strides with a contiguous head dim: "
      "o = softmax(q k^T * softmax_scale) v; head dims (128, 128) or (192, 128). Returns (out, lse or None)",
      py::arg("q"),
      py::arg("k"),
      py::arg("v"),
      py::arg("out") = py::none(),
      py::arg("causal") = false,
      py::arg("softmax_scale") = py::none(),
      py::arg("layout") = "bshd",
      py::arg("return_lse") = false);

  m.def(
      "forward_varlen",
      &forward_varlen,
      "Attention over variable-length sequences, THD (any token/head strides), int32 cu_seqlens on the host "
      "or the GPU. Returns (out, lse or None)",
      py::arg("q"),
      py::arg("k"),
      py::arg("v"),
      py::arg("cu_seqlens_q"),
      py::arg("cu_seqlens_k"),
      py::arg("max_seqlen_q"),
      py::arg("max_seqlen_k"),
      py::arg("out") = py::none(),
      py::arg("causal") = false,
      py::arg("softmax_scale") = py::none(),
      py::arg("return_lse") = false,
      py::arg("host_cu_seqlens_q") = py::none(),
      py::arg("host_cu_seqlens_k") = py::none());

  m.def(
      "forward_paged",
      &forward_paged,
      "Attention over variable-length sequences with K/V in pages [pages, page_size, kv_heads, D]: host "
      "cu_seqlens_q, page_starts and seqlens_k, a GPU page table. Returns (out, lse or None)",
      py::arg("q"),
      py::arg("k_cache"),
      py::arg("v_cache"),
      py::arg("cu_seqlens_q"),
      py::arg("page_table"),
      py::arg("page_starts"),
      py::arg("seqlens_k"),
      py::arg("out") = py::none(),
      py::arg("causal") = false,
      py::arg("softmax_scale") = py::none(),
      py::arg("return_lse") = false);

  py::class_<VarlenPlan>(m, "VarlenPlan")
      .def(py::init<>())
      .def(
          "plan",
          &VarlenPlan::plan,
          py::arg("cu_seqlens_q"),
          py::arg("cu_seqlens_k"),
          py::arg("max_seqlen_q"),
          py::arg("max_seqlen_k"),
          py::arg("total_q"),
          py::arg("total_k"),
          py::arg("heads"),
          py::arg("kv_heads"),
          py::arg("head_dims"),
          py::arg("causal"),
          py::arg("host_cu_seqlens_q") = py::none(),
          py::arg("host_cu_seqlens_k") = py::none())
      .def(
          "run",
          &VarlenPlan::run,
          py::arg("q"),
          py::arg("k"),
          py::arg("v"),
          py::arg("out") = py::none(),
          py::arg("softmax_scale") = py::none(),
          py::arg("return_lse") = false)
      .def("close", &VarlenPlan::close)
      .def_property_readonly("planned", &VarlenPlan::planned)
      .def_property_readonly("causal", &VarlenPlan::causal);
}
