# moonmath-amd

Hand-tuned kernels for AMD CDNA3 (MI300X / gfx942): a bf16 MHA forward kernel, an
MLA (DeepSeek-V3) absorbed-decode kernel, and the grouped GEMMs of an MXFP4
mixture-of-experts layer. And for AMD CDNA4 (MI355X / gfx950): a bf16 attention
forward kernel, [details below](#cdna4-mi355x--gfx950).

**MHA forward** — 8-wave warp-specialized CTA: each wave owns 3 q-tiles (48 q-rows),
two parked in registers, the third staged through LDS; K streams HBM→LDS by direct
DMA and V is consumed pre-transposed straight from L1. Inputs are taken natively in
either `[B, S, H, D]` (BSHD) or `[B, H, S, D]` (BHSD) layout — no transposes anywhere.

A FlashDecoding-style **dense tail KV-split** recovers the stranded fractional
CU-round: when the grid doesn't tile evenly across the 304 CUs, the last partial
round's q-blocks are split along KV across the idle CUs and merged in fp32. It
turns on automatically only when a cost model says it pays (otherwise a single
launch), and is the main reason RTZ now beats AITER on every benchmarked shape.

**MLA absorbed decode** — bf16 Q against fp8 e4m3-fnuz KV. One fused 576-wide KV row
is BOTH K and V (`W_UK`/`W_UV` absorbed outside), so the KV stream is read once at fp8
width and every MFMA is `v_mfma_f32_16x16x16bf16_1k`. One op, `mla_decode_a16w8`, serves
plain decode, speculative-decode verify and decode context parallelism (DCP): any `q_len`,
`H` up to 128, over a device-driven paged pool (`page_size = 1`). It plans from tensor
shapes alone, so it is cuda-graph capturable with no host synchronization, and splits a
batch's KV by request length. Under DCP it also returns a base-2 LSE, which
`mla_dcp_lse_merge_ranks` combines across ranks.

**MXFP4 MoE GEMMs** — `mxfp4_moe_gateup` and `mxfp4_moe_down`, the grouped GEMMs of
an MXFP4 mixture-of-experts layer: E2M1 weight nibbles with one E8M0 scale per 32 k,
bf16 activations. [Details below](#mxfp4-moe-gemms).

## Install

Requires ROCm with `hipcc` on PATH.

```sh
pip install -e .                              # GPU_ARCHS=native picks the GPUs rocminfo reports
GPU_ARCHS="gfx942;gfx950" pip install -e .    # or an explicit list, e.g. on a GPU-less build machine
```

As in AITER, `GPU_ARCHS` picks the targets (`CDNA3_ARCH` still works when it is unset). Each
arch folder in `csrc/` is its own extension module, compiled for that arch only:

- `moonmath_amd._C_gfx942` — the MHA kernel in all three bf16 rounding modes (RTNA, RTNE, RTZ),
  the MLA decode kernel and the MXFP4 MoE GEMMs.
- `moonmath_amd._C_gfx950` — the CDNA4 attention kernels.

Ops pick the module of the GPU they run on at call time, and say so when this build or this GPU
has no kernels for them.

## Use

### MHA forward

```python
import torch
import moonmath_amd as ma

# BSHD tensors as used by diffusion: no transpose required
q = torch.randn(2, 8192, 24, 128, dtype=torch.bfloat16, device="cuda")
k = torch.randn(2, 8192, 24, 128, dtype=torch.bfloat16, device="cuda")
v = torch.randn(2, 8192, 24, 128, dtype=torch.bfloat16, device="cuda")

out      = ma.forward(q, k, v, layout="bshd")                    # gfx942 defaults to RTNA rounding
out_rtna = ma.forward(q, k, v, layout="bshd", round_mode="rtna")
out_rtz  = ma.forward(q, k, v, layout="bshd", round_mode="rtz")

# classic BHSD behaves identically (and is the default layout)
qh = q.transpose(1, 2).contiguous()
out_h = ma.forward(qh, qh, qh)

# cross-attention: arbitrary KV length, no padding needed
ctx = torch.randn(2, 512, 24, 128, dtype=torch.bfloat16, device="cuda")
out_x = ma.forward(q, ctx, ctx, layout="bshd")
```

The kernel runs on the AMD GPU and is launched on the caller's current stream
(no device synchronization, so it overlaps cleanly inside larger pipelines).
CPU tensors are copied to the GPU and back under the hood.

### MLA decode

`q_lat` / `q_pe` are the absorbed latent and RoPE queries, `[B*q_len, H, *]`; the pool holds
ONE fused fp8 row per token (`[..., :512]` latent, `[..., 512:576]` rope) at a single
per-tensor `kv_scale`. The output is written into `o_lat` in place.

```python
import torch
import moonmath_amd as ma

B, H, S, LAT, ROPE = 8, 16, 8192, 512, 64
scale, kv_scale = (LAT + ROPE) ** -0.5, 1.0 / 32.0
pool = (torch.randn(B * S, 1, LAT + ROPE, device="cuda") / kv_scale).to(torch.float8_e4m3fnuz)
kv_indices = torch.arange(B * S, dtype=torch.int32, device="cuda")  # page_size 1; slots in any order
kv_indptr = torch.arange(B + 1, dtype=torch.int32, device="cuda") * S
seq_lens = torch.full((B,), S, dtype=torch.int32, device="cuda")

# q_len 1 is plain decode; q_len 4 is a draft window, position t attending KV [0, S - q_len + t].
for q_len in (1, 4):
    q_lat = torch.randn(B * q_len, H, LAT, dtype=torch.bfloat16, device="cuda")
    q_pe = torch.randn(B * q_len, H, ROPE, dtype=torch.bfloat16, device="cuda")
    o_lat = torch.empty_like(q_lat)
    ma.mla_decode_a16w8(q_lat, q_pe, pool, o_lat, seq_lens, kv_indices, kv_indptr, scale, kv_scale)
```

Under DCP, `seq_lens` counts this rank's positions (`p % cp_world == cp_rank`) and `glen`
the global lengths; pass `lse=` and merge the ranks after the all-to-all:

```python
ma.mla_decode_a16w8(q_lat, q_pe, pool, o_lat, seq_lens, kv_indices, kv_indptr, scale, kv_scale,
                    lse=lse, glen=glen, cp_rank=cp_rank, cp_world=cp_world)
ma.mla_dcp_lse_merge_ranks(parts_in, lse_in, out, lse_out)
```

## CDNA4 (MI355X / gfx950)

bf16 attention forward: head dim 128, and MLA prefill (Q/K 192, V 128). Dense or causal, GQA,
any sequence lengths, as a batch or as packed variable-length sequences. 8-wave CTAs of
16x16x32 MFMAs stream K and V through LDS by DMA; a host planner splits KV across CUs when
the grid would leave them idle. Same `ma.forward`, plus `ma.forward_varlen` in
`flash_attn_varlen_func`'s argument order and `ma.forward_paged` for K/V in pages (AITER's
`mha_batch_prefill_func` page lists, or vLLM block tables).

The kernels index every tensor at its own batch, token and head strides, so any view whose
head dim is contiguous (`stride(-1) == 1`) goes in as is: BSHD or BHSD, slices of a longer
sequence or a larger head set, padded rows, a fused QKV projection, and each of q, k, v and
`out` in its own memory order. Nothing is transposed or copied; `layout` only says which of
dims 1 and 2 is the sequence.

```python
q = torch.randn(2, 4096, 32, 128, dtype=torch.bfloat16, device="cuda")   # shape [B, S, H, D]
k = torch.randn(2, 4096, 8, 128, dtype=torch.bfloat16, device="cuda")    # 8 KV heads, i.e. GQA
v = torch.randn(2, 4096, 8, 128, dtype=torch.bfloat16, device="cuda")
out = ma.forward(q, k, v, layout="bshd", causal=True)   # RTNE rounding, softmax_scale = 1/sqrt(D)

# same data viewed as [B, H, S, D] (the default layout): identical bits, no copies
out_h = ma.forward(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal=True)

# a fused projection [B, S, 32 + 8 + 8, 128]: q, k, v are slices into it
qkv = torch.randn(2, 4096, 48, 128, dtype=torch.bfloat16, device="cuda")
out = ma.forward(qkv[:, :, :32], qkv[:, :, 32:40], qkv[:, :, 40:], layout="bshd", causal=True)

# packed sequences of 1000, 47, 3049 tokens: [total_tokens, H, D] with int32 cu_seqlens
qp, kp, vp = q[0], k[0], v[0]
cu = torch.tensor([0, 1000, 1047, 4096], dtype=torch.int32, device="cuda")
out = ma.forward_varlen(qp, kp, vp, cu, cu, 3049, 3049, causal=True)

# optionally also each row's log-sum-exp: fp32 [B, H, S] (varlen: [H, total_tokens]), matching AITER
out, lse = ma.forward(q, k, v, layout="bshd", causal=True, return_lse=True)

# plan once per forward pass, then run it for each layer
plan = ma.VarlenPlan().plan(cu, cu, 3049, 3049, 4096, 4096, heads=32, kv_heads=8, causal=True)
out = plan.run(qp, kp, vp)

# K/V in 16-token pages: caches [pages, 16, kv_heads, D] with SGLang / AITER page lists
k_cache = torch.randn(512, 16, 8, 128, dtype=torch.bfloat16, device="cuda")
v_cache = torch.randn(512, 16, 8, 128, dtype=torch.bfloat16, device="cuda")
pages = torch.randperm(512, device="cuda")[:257].int()   # pages per sequence: 63 + 3 + 191
out = ma.forward_paged(qp, k_cache, v_cache, cu.cpu(), torch.tensor([1000, 47, 3049], dtype=torch.int32),
                       kv_indptr=torch.tensor([0, 63, 66, 257], dtype=torch.int32), kv_page_indices=pages,
                       causal=True)
```

All run on the current stream and are CUDA-graph capturable. `cu_seqlens` on the GPU are
planned on the GPU: no host sync, and a graph replay reads their current contents. On the host
(CPU tensors), or with CPU copies passed as `host_cu_seqlens_q/_k`, an eager call plans exactly
on the host instead (it can split KV over idle CUs).

`ma.VarlenPlan` splits `forward_varlen` in two: `plan()` takes the sequences and the shapes once
per forward pass, `run(q, k, v)` then launches only the attention kernels, for each layer. Plan
and runs go in the same graph capture or none; keep the plan alive while a graph using it replays.

## Constraints

### MHA forward, gfx942

- bf16 inputs / bf16 outputs.
- `head_dim == 128`.
- Any `seq_len ≥ 1` for Q and K/V independently (cross-attention supported);
  out-of-range rows are handled by hardware buffer bounds, not padding.
- No causal mask, no GQA, no varlen batching; `softmax_scale` fixed at `1/sqrt(128)`.

### Attention forward, gfx950

- bf16 inputs / bf16 outputs, rounded RTNE (`round_mode="rtne"`, the only mode).
- Head dims (Q/K, V) of (128, 128) or (192, 128).
- `ma.forward`: BHSD (`layout="bhsd"`, the default on every arch) or BSHD (`layout="bshd"`);
  `ma.forward_varlen`: THD (`[total_tokens, H, D]`).
- Any strides with a contiguous head dim (`stride(-1) == 1`), tokens at most 2^22 elements
  apart. A new output is contiguous in the call's layout.
- Q heads a multiple of K/V heads (MHA, GQA, MQA).
- Any lengths ≥ 0, Sq ≠ Skv; causal is bottom-right aligned, and a row that sees no key is 0.
- `forward_varlen`: `max_seqlen_q/k` must bound every sequence (checked when the lengths are on
  the host); rows outside every sequence are not written.
- `forward_paged`: page sizes a power of two, at least 8 (head dim 128) or 16 (MLA); head dim
  128 needs K and V caches with the same strides. Lengths and `kv_indptr` are read on the host
  (CPU tensors avoid a sync), so it is not graph capturable; the page list stays on the GPU.

### MLA decode

- bf16 Q (`q_lat` `[.., H, 512]`, `q_pe` `[.., H, 64]`) against fp8 e4m3-fnuz KV;
  bf16 output. `kv_lora_rank = 512`, `qk_rope_head_dim = 64`.
- Fused 576-wide KV rows at ONE per-tensor `kv_scale` — latent and rope share it.
- `H` 1..128 and any `q_len` (one per batch), capped at `B * ceil(q_len * H / 96) ≤ 304`.
- DCP needs `glen` whenever `cp_world > 1`.
- Paged pools are `page_size = 1`, so `kv_indices` is a flat per-token slot list and
  any permutation or subset of it is legal.
- gfx942 / MI300X only (CDNA3).

## Numerics

All three bf16 rounding modes match AITER's per-mode rounding rule. NaN/Inf
handling is bit- and position-identical with AITER for every rounding mode
(canonical `0x7FFF` NaN output), and every finite output element is within
1 bf16 ULP of AITER's. Outputs are deterministic run-to-run.

The MLA decode kernel is deterministic too: its KV split is a function of the inputs and
the fp32 partial merge sums in a fixed association. Against an fp32 reference over
the same dequantized KV (B=2, S=8192, H=16, q_len 4) it lands at **2.5e-3**
relative error, against AITER's own a16w8 asm kernel at 6.3e-3 on those same inputs.
`benchmark/bench_mla.py` prints both before it times anything.

## MXFP4 MoE GEMMs

`mxfp4_moe_gateup` and `mxfp4_moe_down` are the grouped GEMMs of an MXFP4
mixture-of-experts layer. Weights are E2M1 nibbles with one E8M0 scale per 32 k.
Activations stay bf16 and are not quantized.

Gate/up walks K in slabs and takes an optional fused SituGLU epilogue. Down stages
its whole A tile in LDS and sweeps column chunks against it, which costs one barrier
for the GEMM but needs a short K. Down serves K = 384/512/768, the tensor-parallel
shards of the expert width; at any other K `mxfp4_moe_down_block_m` returns 0 and
the down projection runs on the gate/up kernel with `EPI_NONE`.

```python
w13 = ma.repack_mxfp4(w13_raw)          # do this once, when loading weights
w13s = ma.repack_mxfp4_scales(w13s_raw)

bm = ma.mxfp4_moe_gateup_block_m(rows, num_active_experts)
sorted_ids, expert_ids, ntpp, nblk = moe_align_block_size(topk_ids, bm)
ma.mxfp4_moe_gateup(hidden, w13, w13s, out, None, sorted_ids, expert_ids,
                    ntpp, nblk, bm, rows, top_k, epilogue=ma.EPI_SITU)
```

### Results — Kimi-K3 at TP8, MI300X

896 routed experts of width 3072, top-16, K = 3584 (K3's
`routed_expert_hidden_size`, not the model's 7168 `hidden_size`). I = 384 is the TP8
shard. Four shapes: decode at T = 8 and T = 32, which land on 123 and 394 of the 896
experts, and chunked-prefill chunks of 8192 and 16384 tokens, which land on all 896.
Each token draws 16 distinct experts, uniform.

The baseline is a tuned AITER. There is no `gfx942-MOE-MX_FP4.json`, so the bench
times twelve tile candidates at every shape and reports the fastest, aligning the
metadata separately per tile so each pays its own padding. Median of 5 passes,
`fused_moe_mxfp4` from a stock aiter.

| Tokens | Rows | Projection | Ours (ms) | AITER best (ms) | Best tile | Speedup | Ours TFLOP/s |
|---|---|---|---|---|---|---|---|
| 8 | 128 | gate/up | **0.050** | 0.066 | 16x64_k2_e4 | 1.32× | 14 |
| 8 | 128 | down | **0.028** | 0.052 | 16x64_w2_s1 | 1.89× | 13 |
| 32 | 512 | gate/up | **0.172** | 0.240 | 16x64_k2_e4 | 1.39× | 16 |
| 32 | 512 | down | **0.088** | 0.108 | 16x64_w2_s1 | 1.22× | 16 |
| 8192 | 131072 | gate/up | **1.628** | 2.346 | 64x256_s1_k2_e2 | 1.44× | 443 |
| 8192 | 131072 | down | **1.055** | 1.463 | 64x256_s1_k2_e2 | 1.39× | 342 |
| 16384 | 262144 | gate/up | **2.989** | 4.022 | 64x256_s1_k2_e2 | 1.35× | 483 |
| 16384 | 262144 | down | **1.796** | 2.551 | 64x256_s1_k2_e2 | 1.42× | 402 |

Geomean 1.37× gate/up, 1.46× down. At decode, 512 rows over 394 experts is 1.3 rows
per expert, so most of each tile is padding and only the 16-row tile is competitive.

Reproduce with:

```sh
python benchmark/bench_moe.py --markdown            # reproduce the table above
python benchmark/bench_moe.py --tokens 8 512 2048   # any other token count
python benchmark/bench_moe.py --layout ep8          # expert-parallel layout
python benchmark/bench_moe.py --show-tiles          # list all AITER tiles, not only the best
```

The bench needs AITER on the path for the Triton baseline
(`PYTHONPATH=/path/to/aiter`), and `python -m pytest tests/test_moe.py -q` checks
both kernels against a dense reference that never sees the repacked weights.

## Layout / build internals

- `csrc/gfx942/attention_kernel.hip` — the MHA kernel (attention + V pre-transpose).
- `csrc/gfx942/mla_decode_a16w8.hip` — MLA absorbed decode, any q_len, optional DCP.
- `csrc/gfx942/mxfp4_moe_gateup.hip`, `csrc/gfx942/mxfp4_moe_down.hip` — the MoE GEMMs.
- `csrc/gfx950/attention.hip`, `csrc/gfx950/attention_mla.hip` — the CDNA4 attention kernels (head dim
  128, MLA 192/128); `attention_plan.hpp` is their host planner, `attention.h` their C API,
  `opus/` their vendored opus.
- `csrc/<arch>/*_api.cpp` — the torch bindings for each.
- `moonmath_amd/` — Python package; `_arch.py` picks the module of the live GPU.
- `Makefile` — direct kernel build (`make` produces root-level `.so` variants).
- `benchmark/runner.py` — single-shape benchmark vs AITER and (optionally) Modular MAX.
- `benchmark/bench_table.py` — multi-shape sweep with median-over-passes timing.
- `benchmark/bench_mla.py` — MLA decode vs AITER's a16w8 ASM kernel, CUDA-graph timed.
- `benchmark/bench_moe.py` — MXFP4 MoE GEMMs vs AITER's Triton MXFP4 kernel.
- `tests/test_mla_decode.py` — the MLA op against an fp32 reference built from the
  same dequantized KV: every draft length, the end-aligned causal window, the length
  split, DCP ranks, the cross-rank merge, CUDA-graph replay and the domain rejections.
- `tests/test_moe.py` — the MoE GEMMs against a dense reference built from the
  dequantized stock weights, plus the repack, tile-shape and domain contracts.
- `tests/test_attention_gfx950.py` — the CDNA4 kernels against an fp64 reference: dense and
  causal, GQA, off-grid and zero lengths, varlen, varlen == batch bit for bit, BHSD, strided
  views and head-major varlen == their contiguous copies bit for bit, streams and graphs,
  LSE, GPU `cu_seqlens` and `VarlenPlan` (no host sync, graph replays with new lengths,
  == the host-planned path), paged K/V == varlen bit for bit (page sizes 8-64, page lists and
  block tables), and the input rejections. The CDNA3 tests skip on other GPUs, and so does this one.

## Bench

`runner.py` compares `ma.forward` against
[AITER](https://github.com/ROCm/aiter)'s `flash_attn_func` (V3 ASM forward) on
identical BSHD inputs across all three rounding modes. If the
[Modular MAX](https://www.modular.com/max) package is installed it also benches
`max.nn.kernels.flash_attention_gpu`; MAX is loaded and timed only after the
HIP/AITER timings complete so its runtime cannot perturb them.

### Results — MHA forward, MI300X, bf16, head\_dim = 128

Median of 5 independent timing passes (30 iters each) per shape, **with the dense
tail KV-split enabled**. Speedups are `other_ms / ours_ms`, so >1× means we win.
Ours and AITER are a fresh idle-GPU run; Modular MAX figures are carried from the
prior measurement on the same GPU (MAX is kernel-independent — the tail only
affects our column — and its runtime perturbs co-located timings). MAX has no
rounding-mode selector and rounds RTNE internally (verified empirically).

| Shape (B, H, S, D) | Round | Ours (ms) | AITER v3 (ms) | Speedup vs AITER | Modular MAX (ms) | Speedup vs MAX |
|---|---|---|---|---|---|---|
| (2, 24, 8192, 128) | RTNE | **3.083** | 3.792 | 1.23× | 4.237 | 1.37× |
| (2, 24, 8192, 128) | RTNA | **3.022** | 3.605 | 1.19× | 4.237 | 1.40× |
| (2, 24, 8192, 128) | RTZ | **2.983** | 3.303 | 1.11× | 4.237 | 1.42× |
| (2, 24, 16384, 128) | RTNE | **11.670** | 14.691 | 1.26× | 17.923 | 1.54× |
| (2, 24, 16384, 128) | RTNA | **11.479** | 13.801 | 1.20× | 17.923 | 1.56× |
| (2, 24, 16384, 128) | RTZ | **11.385** | 12.629 | 1.11× | 17.923 | 1.57× |
| (1, 32, 16384, 128) | RTNE | **8.013** | 9.031 | 1.13× | 11.030 | 1.38× |
| (1, 32, 16384, 128) | RTNA | **7.828** | 8.656 | 1.11× | 11.030 | 1.41× |
| (1, 32, 16384, 128) | RTZ | **7.731** | 7.989 | 1.03× | 11.030 | 1.43× |
| (4, 16, 16384, 128) | RTNE | **15.591** | 18.337 | 1.18× | 22.061 | 1.41× |
| (4, 16, 16384, 128) | RTNA | **15.331** | 17.567 | 1.15× | 22.061 | 1.44× |
| (4, 16, 16384, 128) | RTZ | **15.055** | 16.183 | 1.07× | 22.061 | 1.47× |
| (1, 64, 16384, 128) | RTNE | **15.528** | 18.333 | 1.18× | 22.763 | 1.47× |
| (1, 64, 16384, 128) | RTNA | **15.239** | 17.535 | 1.15× | 22.763 | 1.49× |
| (1, 64, 16384, 128) | RTZ | **15.040** | 16.161 | 1.07× | 22.763 | 1.51× |
| (2, 24, 32768, 128) | RTNE | **46.002** | 54.794 | 1.19× | 69.947 | 1.52× |
| (2, 24, 32768, 128) | RTNA | **44.440** | 52.363 | 1.18× | 69.947 | 1.57× |
| (2, 24, 32768, 128) | RTZ | **44.075** | 48.549 | 1.10× | 69.947 | 1.59× |
| (2, 16, 65536, 128) | RTNE | **117.612** | 136.301 | 1.16× | 171.273 | 1.46× |
| (2, 16, 65536, 128) | RTNA | **115.550** | 130.278 | 1.13× | 171.273 | 1.48× |
| (2, 16, 65536, 128) | RTZ | **114.665** | 121.668 | 1.06× | 171.273 | 1.49× |
| (2, 8, 86016, 128) | RTNE | **101.071** | 118.939 | 1.18× | 141.319 | 1.40× |
| (2, 8, 86016, 128) | RTNA | **100.165** | 114.515 | 1.14× | 141.319 | 1.41× |
| (2, 8, 86016, 128) | RTZ | **99.397** | 106.513 | 1.07× | 141.319 | 1.42× |
| (1, 16, 131072, 128) | RTNE | **232.517** | 269.278 | 1.16× | 339.322 | 1.46× |
| (1, 16, 131072, 128) | RTNA | **228.475** | 258.092 | 1.13× | 339.322 | 1.49× |
| (1, 16, 131072, 128) | RTZ | **226.152** | 239.587 | 1.06× | 339.322 | 1.50× |


Geomean speedup across shapes:
- **RTNE** — ours **1.18×** vs AITER, **1.44×** vs MAX
- **RTNA** — ours **1.15×** vs AITER, **1.47×** vs MAX
- **RTZ** — ours **1.08×** vs AITER, **1.49×** vs MAX

We now beat AITER on **every shape and every rounding mode**. RTNE/RTNA lead by
1.11–1.26×; RTZ — historically the tightest race, since RTZ is AITER's own fastest
variant — wins 1.03–1.11×. The dense tail KV-split is what erased the prior RTZ
losses at the three 16K B·H ≥ 32 shapes (e.g. (4, 16, 16384) RTZ went 0.95× → 1.07×).
The lead holds with context — 32K through 128K stay 1.06–1.19× across all modes.
Against Modular MAX we are 1.37–1.59× faster everywhere.

Reproduce with:

```sh
# --no-max keeps the ours/AITER numbers cleanest (MAX runtime perturbs co-located timings);
# omit it to measure Modular MAX as well.
python bench_table.py --benchmark-iters 30 --warmup-iters 8 --passes 5 --no-max
```

### Running the bench from scratch

```sh
git clone https://github.com/moonmath-ai/cdna3-attention.git
cd cdna3-attention

# python env (ninja is needed for AITER's JIT and our kernel build)
conda create -n cdna3 python=3.11 ninja -y
conda activate cdna3

# install the package plus bench deps (torch, amd-aiter, numpy; max optional)
pip install -e '.[bench]'

# --- alternatively, with uv ---
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e '.[bench]'

# run it. The first AITER call JIT-builds its fmha modules (~50s, cached afterwards in ~/.aiter/).
python benchmark/runner.py --warmup-iters 8 --benchmark-iters 30
```

`ninja` must be on `$PATH` for AITER's JIT, not just installed — the
conda recipe above takes care of it.

If `max` isn't installed (or you pass `--no-max`), runner skips the MAX row
and prints a one-line "skipped" notice. MAX is initialized only after the
HIP and AITER timing loops have finished, so its runtime cannot perturb them.

### Results — MLA decode, MI300X, bf16 Q / fp8 KV, H = 16, q\_len = 4

`mla_decode_a16w8` against AITER's `a16w8` MLA decode ASM kernel — the only
AITER cell with our dtypes. Both sides run in one process on one shared paged fp8 KV
pool, same Q, same softmax scale, same end-aligned causal mask, same `page_size = 1`
shuffled slot permutation (a kernel that assumed contiguous slots would fail the
validation step). Timed as CUDA-graph replays — AITER's python op wrappers cost
~78 µs/call, which would otherwise swamp the kernel below S ≈ 64K — with
`num_kv_splits` swept per shape and only validated configs kept, and each candidate
timed once per round in alternating order, median over rounds. Speedups are
`aiter_µs / ours_µs`, so >1× means we win.

| Shape (B, S) | KV (MB) | Ours (µs) | AITER a16w8 (µs) | Speedup | Ours (TB/s) | AITER (TB/s) |
|---|---|---|---|---|---|---|
| (1, 150000) | 86 | **76.8** | 133.4 | 1.74× | 1.13 | 0.65 |
| (2, 150000) | 173 | **112.4** | 157.0 | 1.40× | 1.54 | 1.10 |
| (8, 8192) | 38 | **51.8** | 57.7 | 1.12× | 0.73 | 0.65 |
| (8, 32768) | 151 | **101.8** | 133.9 | 1.32× | 1.48 | 1.13 |
| (8, 65536) | 302 | **180.4** | 247.8 | 1.37× | 1.67 | 1.22 |
| (8, 150000) | 691 | **394.2** | 531.6 | 1.35× | 1.75 | 1.30 |
| (16, 150000) | 1382 | **811.1** | 1045.4 | 1.29× | 1.70 | 1.32 |
| (32, 8192) | 151 | **104.1** | 139.2 | 1.34× | 1.45 | 1.08 |

AITER's `a16w8` kernel asserts `nhead == 16`, and rejects q_len > 4 on fp8 KV, so
H = 12 (a DSV3 TP8 shard) and q_len 8 have no like-for-like AITER cell at all. That is
why the script pins H and q_len rather than sweeping them.

Reproduce with:

```sh
# Requires an importable AITER with its gfx942 MLA kernels built (hsa/gfx942/mla/*.co).
# Set AITER_PATH to a checkout if it is not already on sys.path.
python benchmark/bench_mla.py

python benchmark/bench_mla.py --shapes 8:150000,16:150000   # choose shapes
python benchmark/bench_mla.py -v                            # print the num_kv_splits sweep
```

The script validates both kernels against a chunked fp32 streaming-softmax reference
over the same dequantized KV before timing anything, and refuses to rank a config that
produced NaN.

See `examples/basic.py` for a small correctness check using a fp32 reference.
