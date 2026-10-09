// Shared types, geometry and softmax constants for attention.hip and attention_mla.hip.
#pragma once

#include <hip/hip_runtime.h>

namespace attn {

using bf16_t    = __bf16;
using bf16x4_t  = __attribute__((ext_vector_type(4)))  bf16_t;   // ds_read_b64_tr_b16 result
using bf16x8_t  = __attribute__((ext_vector_type(8)))  bf16_t;   // MFMA A/B operand (4 VGPR)
using bf16x16_t = __attribute__((ext_vector_type(16))) bf16_t;   // smfmac dense B operand (8 VGPR)
using f32x4_t   = __attribute__((ext_vector_type(4)))  float;    // MFMA C/D accumulator (4 VGPR)

// ---- geometry constants -------------------------------------------------------------------------
constexpr int WaveSize  = 64;
constexpr int NumWaves  = 8;                       // 2 per SIMD; one CTA per CU
constexpr int BlockSize = NumWaves * WaveSize;     // threads per block, 512
constexpr int KvTile    = 64;                      // KV rows per loop iteration
constexpr int MaxWaveRows = 48;                    // max q rows per wave: three 16-row q tiles, or two (QTiles)
constexpr int BlockM    = NumWaves * MaxWaveRows;  // max q rows per CTA, 384
// v_mfma_f32_16x16x32_bf16 tiling. QK: S^T[kv][q] = K Q^T (M = kv, K = d).
// PV: O^T[d][q] = V^T P^T (M = d, K = kv).
constexpr int KvMTiles = KvTile / 16;              // QK M-tiles: 4 key rows each
constexpr int KvKSteps = KvTile / 32;              // PV k-steps: 2
// One wave's DMA instruction fills one LDS slab: 64 lanes x 16 B.
constexpr int SlabElems = WaveSize * 16 / (int)sizeof(bf16_t);   // 512 elements per slab

// ---- softmax constants --------------------------------------------------------------------------
constexpr float Log2e = 1.4426950408889634f;
// Finite sentinels: -ffast-math enables finite-math-only, making INFINITY UB.
constexpr float NegInf    = -1e30f;
constexpr float MaskScore = -3e30f;                // below NegInf, so a fully masked tile yields P = 0
constexpr float MaxFinite = 3.40282347e38f;        // above this is +inf (or NaN)
// A frame is the offset subtracted from S, in log2 units. Frame0 seeds the first frame (it cancels in O and the
// row sum); if a row's first-tile max is below 0, every frame moves to Frame0 above its row max.
constexpr float Frame0 = 10.0f;
// Lazy rescale: a frame holds while the row max is within RescaleThreshold of it (P <= 2^8), then moves
// RescaleHeadroom above the max.
constexpr float RescaleThreshold = 8.0f;
constexpr float RescaleHeadroom  = 16.0f;

}  // namespace attn
