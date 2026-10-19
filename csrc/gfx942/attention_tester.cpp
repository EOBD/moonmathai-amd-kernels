// Standalone correctness + performance tester for moonmath_attention (CDNA3 / MI300X / gfx942)
// Compiles directly against the HIP kernel source. (no PyTorch)
//
// Usage:  ./attention_tester [batch] [heads] [seq_len] [head_dim] [warmup_iters] [timing_iters] [--force-check]
//         Default: 2 24 16384 128 8 30
//         --force-check to force CPU reference check (normally activated only for small dimensions)
//         with only --force-check, defaults to B=1 H=2 S=256 D=128 warmup=2 iters=5


#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <unistd.h>
#include <vector>


extern "C" {
int launch_v_transpose(
    const void* v, void* v_t, int n_rows, int seq_len_per_head, int heads, int layout, void* stream);
int launch_attention_forward(
    const void* q,
    const void* k,
    const void* v,
    void* out,
    int batch,
    int heads,
    int seq_len,
    int seq_len_kv,
    int head_dim,
    int layout,
    void* workspace,
    void* stream);
size_t tail_workspace_bytes(void);
}


#define HIP_CHECK(call)                                                                    \
  do {                                                                                     \
    hipError_t _e = (call);                                                                \
    if (_e != hipSuccess) {                                                                \
      fprintf(stderr, "HIP error at %s:%d — %s\n", __FILE__, __LINE__, hipGetErrorString(_e)); \
      exit(1);                                                                             \
    }                                                                                      \
  } while (0)

#define KERNEL_CHECK(call)                                                                 \
  do {                                                                                     \
    int _status = (call);                                                                  \
    if (_status != 0) {                                                                    \
      fprintf(stderr, "Kernel launch failed at %s:%d — status %d\n",                    \
              __FILE__, __LINE__, _status);                                                 \
      exit(1);                                                                             \
    }                                                                                      \
  } while (0)

static const char* human_bytes(size_t n) {
  static char buf[32];
  if (n >= 1ULL << 30) snprintf(buf, sizeof(buf), "%.2f GiB", n / (double)(1ULL << 30));
  else if (n >= 1ULL << 20) snprintf(buf, sizeof(buf), "%.2f MiB", n / (double)(1ULL << 20));
  else if (n >= 1ULL << 10) snprintf(buf, sizeof(buf), "%.2f KiB", n / (double)(1ULL << 10));
  else snprintf(buf, sizeof(buf), "%zu B", n);
  return buf;
}

static void fill_random_bf16(std::vector<__hip_bfloat16>& buf, unsigned seed) {
  // Fill with N(0, 0.5) then convert to bf16 (truncation).
  std::mt19937 rng(seed);
  std::normal_distribution<float> dist(0.0f, 0.5f);
  for (auto& v : buf) {
    float f = dist(rng);
    v = __float2bfloat16(f);
  }
}

static void attention_ref(
    const std::vector<__hip_bfloat16>& q,
    const std::vector<__hip_bfloat16>& k,
    const std::vector<__hip_bfloat16>& v,
    std::vector<float>& out,
    int B, int H, int S, int D) {
  const float scale = 1.0f / std::sqrt((float)D);
  const size_t head_stride = (size_t)S * D;
  const size_t batch_stride = (size_t)H * head_stride;

  for (int b = 0; b < B; ++b) {
    for (int h = 0; h < H; ++h) {
      const size_t base = (size_t)b * batch_stride + (size_t)h * head_stride;
      const __hip_bfloat16* qh = q.data() + base;
      const __hip_bfloat16* kh = k.data() + base;
      const __hip_bfloat16* vh = v.data() + base;
      float* oh = out.data() + base;
      std::vector<float> scores(S);

      for (int i = 0; i < S; ++i) {
        const __hip_bfloat16* qi = qh + (size_t)i * D;
        float row_max = std::numeric_limits<float>::lowest();
        for (int j = 0; j < S; ++j) {
          const __hip_bfloat16* kj = kh + (size_t)j * D;
          float dot = 0.0f;
          for (int d = 0; d < D; ++d) dot += (float)qi[d] * (float)kj[d];
          scores[j] = dot * scale;
          row_max = std::max(row_max, scores[j]);
        }

        float denom = 0.0f;
        for (int j = 0; j < S; ++j) {
          scores[j] = std::exp(scores[j] - row_max);
          denom += scores[j];
        }

        float* oi = oh + (size_t)i * D;
        std::fill(oi, oi + D, 0.0f);
        const float inv_denom = 1.0f / denom;
        for (int j = 0; j < S; ++j) {
          const float p = scores[j] * inv_denom;
          const __hip_bfloat16* vj = vh + (size_t)j * D;
          for (int d = 0; d < D; ++d) oi[d] += p * (float)vj[d];
        }
      }
    }
  }
}

struct CorrectnessResult {
  bool passed = true;
  double max_abs = 0.0;
  double max_rel = 0.0;
  double rel_l2 = 0.0;
  size_t mismatches = 0;
};

static CorrectnessResult compare_outputs(
    const std::vector<__hip_bfloat16>& actual,
    const std::vector<float>& expected,
    double rel_l2_tol) {
  CorrectnessResult result;
  double squared_error = 0.0;
  double squared_ref = 0.0;

  for (size_t i = 0; i < expected.size(); ++i) {
    const double a = (float)actual[i];
    const double e = expected[i];
    uint16_t actual_bits = 0;
    static_assert(sizeof(actual[i]) == sizeof(actual_bits));
    std::memcpy(&actual_bits, &actual[i], sizeof(actual_bits));
    const bool non_finite = (actual_bits & 0x7f80u) == 0x7f80u;
    const double abs_err = std::fabs(a - e);
    const double rel_err = abs_err / (std::fabs(e) + 1e-6);
    result.max_abs = std::max(result.max_abs, abs_err);
    result.max_rel = std::max(result.max_rel, rel_err);
    squared_error += abs_err * abs_err;
    squared_ref += e * e;

    if (non_finite || (abs_err > 1e-2 && rel_err > 2e-2)) {
      if (result.mismatches < 10) {
        fprintf(stderr,
                "Mismatch at idx=%zu: device=%.6f ref=%.6f abs=%.6f rel=%.6f\n",
                i, a, e, abs_err, rel_err);
      }
      ++result.mismatches;
    }
  }

  result.rel_l2 = squared_ref > 0.0 ? std::sqrt(squared_error / squared_ref)
                                    : std::sqrt(squared_error);
  result.passed = result.mismatches == 0 && result.rel_l2 <= rel_l2_tol;
  return result;
}

struct BenchmarkResult {
  float best_ms;
  float avg_ms;
  double best_tflops;
  double avg_tflops;
  int iters;
  bool correctness_checked;
  bool correctness_passed;
};


constexpr int ROTATING_BUFFERS = 4;  // rotate input buffers to kill cache

BenchmarkResult run_benchmark(int B, int H, int S, int D, int warmup, int iters, bool do_check) {
  if (D != 128) {
    fprintf(stderr, "head_dim must be 128 (got %d)\n", D);
    exit(1);
  }
  if (S % 64 != 0) {
    // pad to next multiple of 64 for V_t transpose
    int s_orig = S;
    S = ((S + 63) / 64) * 64;
    fprintf(stderr, "Note: padded seq_len from %d → %d (64-row boundary for V_t)\n", s_orig, S);
  }

  const size_t elem_count = (size_t)B * H * S * D;
  const size_t byte_count = elem_count * sizeof(__hip_bfloat16);
  const size_t out_count = (size_t)B * H * S * D;
  const size_t out_bytes = out_count * sizeof(__hip_bfloat16);
  const size_t vt_bytes = byte_count;  // V_t same size as V
  const size_t n_rows = B * H * S;

  fprintf(stderr, "[dbg] elem_count=%zu, byte_count=%s, out_bytes=%s, n_rows=%zu\n",
          elem_count, human_bytes(byte_count), human_bytes(out_bytes), n_rows);
  fflush(stderr);

  // Tail workspace: dense-mode split recovery buffer. Same size for all round modes.
  fprintf(stderr, "[dbg] calling tail_workspace_bytes()...\n"); fflush(stderr);
  const size_t workspace_bytes = tail_workspace_bytes();
  fprintf(stderr, "[dbg] tail_workspace_bytes → %s\n", human_bytes(workspace_bytes)); fflush(stderr);
  const bool no_tail = (std::getenv("MOONMATH_NO_TAIL") != nullptr);

  printf("  [mem] Q/K/V: %d×bufs of %s each, V_t: %s, output: %s\n",
         ROTATING_BUFFERS, human_bytes(byte_count), human_bytes(vt_bytes), human_bytes(out_bytes));
  if (!no_tail)
    printf("  [mem] tail workspace: %s\n", human_bytes(workspace_bytes));
  else
    printf("  [mem] tail workspace: disabled (MOONMATH_NO_TAIL=1)\n");

  __hip_bfloat16 *d_q[ROTATING_BUFFERS], *d_k[ROTATING_BUFFERS], *d_v[ROTATING_BUFFERS], *d_vt = nullptr;
  __hip_bfloat16* d_out = nullptr;
  float* d_workspace = nullptr;

  printf("  [mem] allocating GPU buffers...\n"); fflush(stdout);
  fprintf(stderr, "[dbg] starting hipMalloc loop (%d bufs of %s each)...\n",
          ROTATING_BUFFERS, human_bytes(byte_count)); fflush(stderr);
  for (int i = 0; i < ROTATING_BUFFERS; i++) {
    fprintf(stderr, "[dbg]   hipMalloc d_q[%d]...\n", i); fflush(stderr);
    HIP_CHECK(hipMalloc(&d_q[i], byte_count));
    fprintf(stderr, "[dbg]   hipMalloc d_k[%d]...\n", i); fflush(stderr);
    HIP_CHECK(hipMalloc(&d_k[i], byte_count));
    fprintf(stderr, "[dbg]   hipMalloc d_v[%d]...\n", i); fflush(stderr);
    HIP_CHECK(hipMalloc(&d_v[i], byte_count));
  }
  fprintf(stderr, "[dbg] hipMalloc d_vt (%s)...\n", human_bytes(vt_bytes)); fflush(stderr);
  HIP_CHECK(hipMalloc(&d_vt, vt_bytes));
  fprintf(stderr, "[dbg] hipMalloc d_out (%s)...\n", human_bytes(out_bytes)); fflush(stderr);
  HIP_CHECK(hipMalloc(&d_out, out_bytes));
  if (!no_tail) {
    fprintf(stderr, "[dbg] hipMalloc d_workspace (%s)...\n", human_bytes(workspace_bytes)); fflush(stderr);
    HIP_CHECK(hipMalloc(&d_workspace, workspace_bytes));
  }
  fprintf(stderr, "[dbg] all hipMalloc calls done.\n"); fflush(stderr);

  // Fill host buffers and upload (different seed per rotation buffer)
  fprintf(stderr, "[dbg] creating host buffer, elem_count=%zu (%s)...\n",
          elem_count, human_bytes(elem_count * sizeof(__hip_bfloat16))); fflush(stderr);
  std::vector<__hip_bfloat16> h_buf(elem_count);
  fprintf(stderr, "[dbg] host buffer allocated.\n"); fflush(stderr);
  for (int i = 0; i < ROTATING_BUFFERS; i++) {
    printf("  [data] uploading buffer %d/%d (seed=%u)...\n", i + 1, ROTATING_BUFFERS, 42u + i); fflush(stdout);
    fprintf(stderr, "[dbg]   fill_random_bf16 seed=%u...\n", 42u + i); fflush(stderr);
    fill_random_bf16(h_buf, 42u + i);
    fprintf(stderr, "[dbg]   hipMemcpy d_q[%d] (%s)...\n", i, human_bytes(byte_count)); fflush(stderr);
    HIP_CHECK(hipMemcpy(d_q[i], h_buf.data(), byte_count, hipMemcpyHostToDevice));
    fprintf(stderr, "[dbg]   fill_random_bf16 seed=%u...\n", 100u + i); fflush(stderr);
    fill_random_bf16(h_buf, 100u + i);
    fprintf(stderr, "[dbg]   hipMemcpy d_k[%d]...\n", i); fflush(stderr);
    HIP_CHECK(hipMemcpy(d_k[i], h_buf.data(), byte_count, hipMemcpyHostToDevice));
    fprintf(stderr, "[dbg]   fill_random_bf16 seed=%u...\n", 200u + i); fflush(stderr);
    fill_random_bf16(h_buf, 200u + i);
    fprintf(stderr, "[dbg]   hipMemcpy d_v[%d]...\n", i); fflush(stderr);
    HIP_CHECK(hipMemcpy(d_v[i], h_buf.data(), byte_count, hipMemcpyHostToDevice));
    printf("  [data] buffer %d/%d done.\n", i + 1, ROTATING_BUFFERS); fflush(stdout);
  }

  // warmup 
  fprintf(stderr, "[dbg] hipStreamCreate...\n"); fflush(stderr);
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  fprintf(stderr, "[dbg] stream created.\n"); fflush(stderr);

  printf("\n  [phase] WARMUP (%d iterations) ", warmup); fflush(stdout);
  int dot_step = warmup / 80 + 1;  // ~80 dots total
  for (int i = 0; i < warmup; i++) {
    int idx = i % ROTATING_BUFFERS;
    if (i == 0) {
      fprintf(stderr, "[dbg] first warmup launch (v_transpose + fwd)...\n"); fflush(stderr);
    }
    KERNEL_CHECK(launch_v_transpose(
        d_v[idx], d_vt, (int)n_rows, S, H, 0 /*layout=bhsd*/, (void*)stream));
    KERNEL_CHECK(launch_attention_forward(
        d_q[idx], d_k[idx], d_vt, d_out, B, H, S, S, D, 0 /*layout=bhsd*/,
        (void*)(no_tail ? nullptr : d_workspace), (void*)stream));
    fprintf(stderr, "[dbg] warmup iter %d: hipStreamSynchronize...\n", i); fflush(stderr);
    HIP_CHECK(hipStreamSynchronize(stream));
    if ((i + 1) % dot_step == 0) { printf("."); fflush(stdout); }
  }
  printf(" done.\n\n");

  //timed iterations
  hipEvent_t t0, t1;
  HIP_CHECK(hipEventCreate(&t0));
  HIP_CHECK(hipEventCreate(&t1));

  std::vector<float> times;
  times.reserve(iters);

  printf("  [phase] TIMING (%d iterations)\n", iters);
  dot_step = iters / 80 + 1;
  int last_idx = 0;
  for (int r = 0; r < iters; r++) {
    int idx = r % ROTATING_BUFFERS;
    last_idx = idx;
    HIP_CHECK(hipEventRecord(t0, stream));
    KERNEL_CHECK(launch_v_transpose(d_v[idx], d_vt, (int)n_rows, S, H, 0, (void*)stream));
    KERNEL_CHECK(launch_attention_forward(
        d_q[idx], d_k[idx], d_vt, d_out, B, H, S, S, D, 0,
        (void*)(no_tail ? nullptr : d_workspace), (void*)stream));
    HIP_CHECK(hipEventRecord(t1, stream));
    HIP_CHECK(hipEventSynchronize(t1));
    float ms = 0;
    HIP_CHECK(hipEventElapsedTime(&ms, t0, t1));
    times.push_back(ms);
    if ((r + 1) % dot_step == 0) { printf("."); fflush(stdout); }

    int log_step = iters <= 10 ? 1 : (iters / 5);
    if ((r + 1) % log_step == 0 || r == 0)
      printf("  iter %3d: %7.3f ms\n", r + 1, ms);
  }
  printf("  [phase] TIMING done.\n\n");

  HIP_CHECK(hipEventDestroy(t0));
  HIP_CHECK(hipEventDestroy(t1));

  float best = *std::min_element(times.begin(), times.end());
  double sum = 0;
  for (float t : times) sum += t;
  float avg = (float)(sum / times.size());

  // FLOPs: 4 * B * H * S^2 * D (QK + PV matmuls)
  double flops = 4.0 * (double)B * H * S * S * D;

  bool correctness_passed = true;
  if (do_check) {
    printf("  [phase] CORRECTNESS (CPU reference)\n");
    std::vector<__hip_bfloat16> h_out(out_count);
    HIP_CHECK(hipMemcpy(h_out.data(), d_out, out_bytes, hipMemcpyDeviceToHost));

    std::vector<__hip_bfloat16> h_q(elem_count), h_k(elem_count), h_v(elem_count);
    fill_random_bf16(h_q, 42u + last_idx);
    fill_random_bf16(h_k, 100u + last_idx);
    fill_random_bf16(h_v, 200u + last_idx);
    std::vector<float> h_ref(out_count);
    attention_ref(h_q, h_k, h_v, h_ref, B, H, S, D);

    constexpr double REL_L2_TOL = 2e-2;
    CorrectnessResult check = compare_outputs(h_out, h_ref, REL_L2_TOL);
    correctness_passed = check.passed;
    printf("  max abs error : %.6f\n", check.max_abs);
    printf("  max rel error : %.6f\n", check.max_rel);
    printf("  relative L2   : %.6f (tol %.3f)\n", check.rel_l2, REL_L2_TOL);
    printf("  mismatches    : %zu / %zu\n", check.mismatches, out_count);
    printf("  Correctness   : %s\n\n", check.passed ? "PASSED" : "FAILED");
  }

  HIP_CHECK(hipStreamDestroy(stream));

  printf("  [mem] freeing GPU buffers..."); fflush(stdout);
  for (int i = 0; i < ROTATING_BUFFERS; i++) {
    HIP_CHECK(hipFree(d_q[i]));
    HIP_CHECK(hipFree(d_k[i]));
    HIP_CHECK(hipFree(d_v[i]));
  }
  HIP_CHECK(hipFree(d_vt));
  HIP_CHECK(hipFree(d_out));
  if (d_workspace) HIP_CHECK(hipFree(d_workspace));
  printf(" done.\n");

  return {best, avg, flops / (best * 1e-3) / 1e12, flops / (avg * 1e-3) / 1e12,
          iters, do_check, correctness_passed};
}


int main(int argc, char** argv) {
  fprintf(stderr, "[dbg] main() entered, pid=%d\n", getpid());
  fflush(stderr);

  int B = 2, H = 24, S = 16384, D = 128;
  int warmup = 10, iters = 30;
  bool force_check = false;

  fprintf(stderr, "[dbg] parsing args...\n"); fflush(stderr);
  std::vector<std::string> positional;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--force-check") force_check = true;
    else positional.emplace_back(argv[i]);
  }
  if (positional.size() > 0) B = atoi(positional[0].c_str());
  if (positional.size() > 1) H = atoi(positional[1].c_str());
  if (positional.size() > 2) S = atoi(positional[2].c_str());
  if (positional.size() > 3) D = atoi(positional[3].c_str());
  if (positional.size() > 4) warmup = atoi(positional[4].c_str());
  if (positional.size() > 5) iters = atoi(positional[5].c_str());
  if (positional.size() > 6) {
    fprintf(stderr, "Too many positional arguments.\n");
    return 2;
  }
  if (force_check && positional.empty()) {
    B = 1; H = 2; S = 256; D = 128; warmup = 2; iters = 5;
  }
  if (B <= 0 || H <= 0 || S <= 0 || warmup < 0 || iters <= 0) {
    fprintf(stderr, "batch, heads, seq_len, and timing_iters must be positive; warmup must be non-negative.\n");
    return 2;
  }
  const size_t check_work = (size_t)B * H * S * S * D;
  const size_t check_threshold = (size_t)1 * 2 * 512 * 512 * 128;
  const bool do_check = force_check || check_work <= check_threshold;
  const char* check_mode = force_check ? "ENABLED (--force-check)"
                           : do_check   ? "ENABLED (auto: small config)"
                                        : "DISABLED (performance only)";
  fprintf(stderr, "[dbg] args: B=%d H=%d S=%d D=%d warmup=%d iters=%d correctness=%s\n",
          B, H, S, D, warmup, iters, do_check ? "enabled" : "disabled");
  fflush(stderr);

  fprintf(stderr, "[dbg] calling hipGetDeviceCount...\n"); fflush(stderr);
  int dev_count;
  HIP_CHECK(hipGetDeviceCount(&dev_count));
  fprintf(stderr, "[dbg] hipGetDeviceCount → %d\n", dev_count); fflush(stderr);
  if (dev_count == 0) {
    fprintf(stderr, "No HIP devices found.\n");
    return 1;
  }

  fprintf(stderr, "[dbg] calling hipGetDeviceProperties(0)...\n"); fflush(stderr);
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  fprintf(stderr, "[dbg] device 0: %s (%s), %zu MB VRAM\n",
          prop.name, prop.gcnArchName, prop.totalGlobalMem / (1024*1024));
  fflush(stderr);

  fprintf(stderr, "[dbg] calling hipSetDevice(0)...\n"); fflush(stderr);
  HIP_CHECK(hipSetDevice(0));
  fprintf(stderr, "[dbg] device set.\n"); fflush(stderr);

  printf("╔══════════════════════════════════════════════════════════════╗\n");
  printf("║   moonmath_attention — correctness + performance tester     ║\n");
  printf("╠══════════════════════════════════════════════════════════════╣\n");
  printf("║   Device  : %-48s ║\n", prop.name);
  printf("║   Arch    : %-48s ║\n", prop.gcnArchName);
  printf("║   Config  : B=%-4d H=%-4d S=%-6d D=%-3d (bhsd, RTZ)   ║\n", B, H, S, D);
  printf("║   Warmup  : %-4d   Timed iters: %-4d                    ║\n", warmup, iters);
  printf("║   Buffers : %d (rotating)                                   ║\n", ROTATING_BUFFERS);
  printf("║   Check   : %-48s ║\n", check_mode);
  printf("╚══════════════════════════════════════════════════════════════╝\n\n");
  fflush(stdout);

  double flops_total = 4.0 * (double)B * H * S * S * D;
  printf("  FLOPs / launch : %.2f GFLOP\n\n", flops_total / 1e9);
  fflush(stdout);

  fprintf(stderr, "[dbg] entering run_benchmark...\n"); fflush(stderr);
  auto res = run_benchmark(B, H, S, D, warmup, iters, do_check);
  fprintf(stderr, "[dbg] run_benchmark returned.\n"); fflush(stderr);

  printf("╔══════════════════════════════════════════════╗\n");
  printf("║              RESULTS                         ║\n");
  printf("╠══════════════════════════════════════════════╣\n");
  printf("║  Best : %8.3f ms   |  %6.2f TFLOPS/s   ║\n", res.best_ms, res.best_tflops);
  printf("║  Avg  : %8.3f ms   |  %6.2f TFLOPS/s   ║\n", res.avg_ms, res.avg_tflops);
  printf("║  Iters: %8d                    ║\n", res.iters);
  if (res.correctness_checked)
    printf("║  Check: %-8s                    ║\n", res.correctness_passed ? "PASSED" : "FAILED");
  else
    printf("║  Check: %-8s                    ║\n", "SKIPPED");
  printf("╚══════════════════════════════════════════════╝\n");

  fprintf(stderr, "[dbg] main() exiting normally.\n"); fflush(stderr);
  return res.correctness_passed ? 0 : 1;
}
