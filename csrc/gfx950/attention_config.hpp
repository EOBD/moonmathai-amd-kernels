// Types, shared geometry and softmax constants for the gfx950 bf16 attention kernels
// (attention.hip, attention_mla.hip); each kernel adds its own head dims and LDS layout.
#pragma once

#include <hip/hip_runtime.h>

namespace attn {

using bf16_t    = __bf16;
using bf16x4_t  = __attribute__((ext_vector_type(4)))  bf16_t;   // result of ds_read_b64_tr_b16
using bf16x8_t  = __attribute__((ext_vector_type(8)))  bf16_t;   // MFMA A/B operand (4 VGPR)
using bf16x16_t = __attribute__((ext_vector_type(16))) bf16_t;   // smfmac dense B operand (8 VGPR)
using f32x4_t   = __attribute__((ext_vector_type(4)))  float;    // MFMA C/D accumulator (4 VGPR)

// ---- geometry constants -------------------------------------------------------------------------
constexpr int WaveSize  = 64;
constexpr int NumWaves  = 8;                       // 2 per SIMD; one CTA per CU
constexpr int BlockSize = NumWaves * WaveSize;     // block size, 512
constexpr int KvTile    = 64;                      // KV rows handled per loop iteration
constexpr int MaxWaveRows = 48;                    // max q rows per wave: three 16-row q tiles, or two (QTiles)
constexpr int BlockM    = NumWaves * MaxWaveRows;  // at most 384 q rows per CTA
// v_mfma_f32_16x16x32_bf16 tiling. QK: S^T[kv][q] = K Q^T (M = kv, K = d).
// PV: O^T[d][q] = V^T P^T (M = d, K = kv).
constexpr int KvMTiles = KvTile / 16;              // 4 QK M-tiles
constexpr int KvKSteps = KvTile / 32;              // 2 PV k-steps
// One wave's DMA instruction fills one LDS slab: 64 lanes x 16 B.
constexpr int SlabElems = WaveSize * 16 / (int)sizeof(bf16_t);   // slab elements, 512

// ---- softmax constants --------------------------------------------------------------------------
constexpr float Log2e = 1.4426950408889634f;
// Finite sentinels: -ffast-math enables finite-math-only, making INFINITY UB.
constexpr float NegInf    = -1e30f;
constexpr float MaskScore = -3e30f;                // below NegInf, so a fully masked tile yields P = 0
constexpr float MaxFinite = 3.40282347e38f;        // beyond it lies +inf (or NaN)
// Frames (attention.hip), in log2 units, start at Frame0; when a row's first-tile max drops below 0,
// every row's frame shifts to Frame0 above its max. The offset cancels in O / row sum, saving MFMA energy.
constexpr float Frame0 = 10.0f;
// Lazy rescale: a frame holds while the row max stays within RescaleThreshold of it (P <= 2^8, the
// bound used by the lazy competitors gluon fmha_v4 and FlyDSL), then jumps RescaleHeadroom above the max.
constexpr float RescaleThreshold = 8.0f;
constexpr float RescaleHeadroom  = 16.0f;

}  // attn namespace
