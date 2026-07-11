#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstdint>

#include "opus.hpp"

namespace mmai_attention {

// ===== BUILDING BLOCKS: types, constants, intrinsics, layouts, tail scratch

using bf16_t = __hip_bfloat16;
using bf16x4_t = short __attribute__((ext_vector_type(4))); // 4 bf16 = 8 bytes
// bf16 rounding selected at compile time via -DBF16_ROUND.
#ifndef BF16_ROUND
#define BF16_ROUND RTZ
#endif
enum class BF16Round { RTNE, RTZ, RTNA };
constexpr BF16Round RoundMode = BF16Round::BF16_ROUND;
using fp32x4_t = float __attribute__((ext_vector_type(4))); // 16x16x16 C-fragment
using u32x2_t = uint32_t __attribute__((ext_vector_type(2)));
using u32x4_t = uint32_t __attribute__((ext_vector_type(4)));

constexpr int WaveSize = 64;
constexpr int NumWaves = 8;
constexpr int LoadWaves = 4;
constexpr int BlockSize = WaveSize * NumWaves;
constexpr int KvBlockRows = 64;                    // kv rows processed per main-loop iter
constexpr int MfmaTile = 16;                       // 16x16x16 mfma tile (M=N)
constexpr int KvSubtiles = KvBlockRows / MfmaTile; // 4 kv-subtiles / block
constexpr int MfmaK = 16;                          // mfma K (contraction)
constexpr int HeadDim = 128;
constexpr int HeadDimTiles = HeadDim / MfmaK;             // 8 head-dim tiles
constexpr int QTileRows = 16;                             // rows per q-tile (one mfma)
constexpr int QTilesPerWave = 3;                          // → 48 q-rows / wave (3 q-tiles)
constexpr int QTilesPerWG = NumWaves * QTilesPerWave;     // 24 q-tiles = 384 rows / workgroup
constexpr int KRowDwords = HeadDim / 2;                   // 64 = real K row width (dwords)
constexpr int KLdsStrideDwords = KRowDwords;              // 64 = LDS row stride (XOR-swizzled, no pad)
constexpr int KRowsPerLoadWave = KvBlockRows / LoadWaves; // 16
constexpr float Log2e = 1.4426950408889634f;
constexpr uint32_t BufferConfig = 0x00020000u; // SRD word3: DATA_FORMAT=32 (raw 32-bit elems)

// Rescale gate: rescale o_acc only when block raises running max by >RescaleThreshold (log2). DEFAULT 0 =
// exact max → rmse ~6e-5, perf-neutral (rescale hides in P2). Override -DRESCALE_THRESHOLD=8.0f if softmax critical.
#ifndef RESCALE_THRESHOLD
#define RESCALE_THRESHOLD 0.0f
#endif
constexpr float RescaleThreshold = RESCALE_THRESHOLD;

// ─── MFMA + scalar math ──────────────────────────────────────────────────────
__device__ __forceinline__ fp32x4_t zero_fp32x4() { return fp32x4_t{0, 0, 0, 0}; }

__device__ __forceinline__ float fast_exp2(float x) { return __builtin_amdgcn_exp2f(x); }
__device__ __forceinline__ float fast_rcp(float x) { return __builtin_amdgcn_rcpf(x); }

// Inline-asm mfma: forces in/out same VGPR (no v_mov copies) and keeps issue in source order.
__device__ __forceinline__ void asm_mfma(bf16x4_t a, bf16x4_t b, fp32x4_t &c) { asm volatile("v_mfma_f32_16x16x16_bf16 %0, %1, %2, %0" : "+v"(c) : "v"(a), "v"(b)); }

// Literal-0 C operand skips the v_mov zero-init; "=&v" prevents aliasing dst onto A VGPRs.
__device__ __forceinline__ fp32x4_t asm_mfma_zero(bf16x4_t a, bf16x4_t b) {
    fp32x4_t r;
    asm volatile("v_mfma_f32_16x16x16_bf16 %0, %1, %2, 0" : "=&v"(r) : "v"(a), "v"(b));
    return r;
}

// Scalar v_add/v_mul: avoids v_pk_*_f32 sharing writeback ports with MFMA on the same wave-pair.
__device__ __forceinline__ float vadd_f32(float a, float b) {
    float r;
    asm volatile("v_add_f32_e32 %0, %1, %2" : "=v"(r) : "v"(a), "v"(b));
    return r;
}
__device__ __forceinline__ float vfma_f32(float a, float b, float c) { // a*b + c
    float r;
    asm volatile("v_fma_f32 %0, %1, %2, %3" : "=v"(r) : "v"(a), "v"(b), "v"(c));
    return r;
}
__device__ __forceinline__ bool wave_all(bool p) { return __all((int)p) != 0; }

// ─── bf16 rounding + pack ────────────────────────────────────────────────────
// bf16 packing. MaybeNaN=true guards 0×inf NaN; guard left to compiler (pinned asm spills ~50 VGPR past 256 budget).
template <bool MaybeNaN> __device__ __forceinline__ unsigned bf16_round_one(float f) {
    if constexpr (RoundMode == BF16Round::RTNE) {
        if constexpr (MaybeNaN) {
            uint32_t u_bits = __builtin_bit_cast(uint32_t, f);
            uint32_t tmp;
            u32x2_t nan_mask;
            asm volatile("v_cmp_u_f32 %0, %2, %2\n"
                         "v_bfe_u32 %1, %2, 16, 1\n"
                         "v_add3_u32 %1, %2, %1, %3\n"
                         "v_cndmask_b32 %2, %1, %4, %0"
                         : "=s"(nan_mask), "=&v"(tmp), "+v"(u_bits)
                         : "v"(0x7FFFu), "v"(0x7FFF0000u));
            return u_bits;
        } else {
            const unsigned u = __builtin_bit_cast(unsigned, f);
            return u + 0x7FFFu + ((u >> 16) & 1u);
        }
    } else if constexpr (RoundMode == BF16Round::RTNA) {
        unsigned biased = __builtin_bit_cast(unsigned, f) + 0x8000u; // ties-away rounding bias
        if constexpr (MaybeNaN) {
            // NaN → canonical 0x7FFF0000. MUST be inline asm: -ffast-math folds __builtin_isunordered to false,
            // disabling a C guard → RTNA leaks biased NaN 0xFFC0 instead of 0x7FFF.
            u32x2_t nan_mask;
            asm volatile("v_cmp_u_f32 %0, %2, %2\n"
                         "v_cndmask_b32 %1, %1, %3, %0"
                         : "=s"(nan_mask), "+v"(biased)
                         : "v"(f), "v"(0x7FFF0000u));
        }
        return biased;
    } else {
        return __builtin_bit_cast(unsigned, f); // RTZ — top 16 bits
    }
}
template <bool MaybeNaN = false> __device__ __forceinline__ bf16x4_t pack4_bf16(float a, float b, float c, float d) {
    const unsigned ua = bf16_round_one<MaybeNaN>(a), ub = bf16_round_one<MaybeNaN>(b);
    const unsigned uc = bf16_round_one<MaybeNaN>(c), ud = bf16_round_one<MaybeNaN>(d);
    // perm 0x07060302 keeps bytes [3,2] (upper 16b = bf16) of each source, packs 2 bf16/dword: w0=[bf16(b)|bf16(a)].
    const unsigned w0 = __builtin_amdgcn_perm(ub, ua, 0x07060302u);
    const unsigned w1 = __builtin_amdgcn_perm(ud, uc, 0x07060302u);
    bf16x4_t o;
    __builtin_memcpy(&o, &w0, 4);
    __builtin_memcpy((char *)&o + 4, &w1, 4);
    return o;
}

// ─── Buffer / LDS load-store ─────────────────────────────────────────────────
// Wrap a global ptr in a 128-bit SRD ("V#"): stride=0 (unstructured), num_records=0xFFFFFFFF (unbounded),
// config=BufferConfig. const_cast: builtin takes non-const base, but a load resource is never written.
__device__ __forceinline__ __amdgpu_buffer_rsrc_t make_buffer(const void *p, uint32_t num_records = 0xFFFFFFFFu) {
    return __builtin_amdgcn_make_buffer_rsrc(const_cast<void *>(p), 0, num_records, BufferConfig);
}
// Raw buffer load/store, named by the value moved (bf16x4 = 8 B, u32x4 = 16 B).
__device__ __forceinline__ bf16x4_t load_bf16x4(__amdgpu_buffer_rsrc_t buffer, int byte_off) {
    u32x2_t raw = __builtin_amdgcn_raw_buffer_load_b64(buffer, byte_off, 0, 0);
    bf16x4_t out;
    __builtin_memcpy(&out, &raw, sizeof(out));
    return out;
}
__device__ __forceinline__ u32x4_t load_u32x4(__amdgpu_buffer_rsrc_t buffer, int byte_off) { return __builtin_amdgcn_raw_buffer_load_b128(buffer, byte_off, 0, 0); }
__device__ __forceinline__ void store_bf16x4(__amdgpu_buffer_rsrc_t buffer, int byte_off, bf16x4_t v) {
    u32x2_t bits;
    __builtin_memcpy(&bits, &v, 8);
    __builtin_amdgcn_raw_buffer_store_b64(bits, buffer, byte_off, 0, 0);
}
// Split a u32x4 (one b128 K/V load = d-tile pair) into 2 bf16x4 MFMA operands: lo2=dwords 0,1; hi2=dwords 2,3.
__device__ __forceinline__ bf16x4_t lo2(u32x4_t v) { return __builtin_bit_cast(bf16x4_t, u32x2_t{v[0], v[1]}); }
__device__ __forceinline__ bf16x4_t hi2(u32x4_t v) { return __builtin_bit_cast(bf16x4_t, u32x2_t{v[2], v[3]}); }

// Low 32 bits of an LDS pointer = its byte offset within the LDS segment.
template <typename T> __device__ __forceinline__ uint32_t lds_addr(const T *p) { return (uint32_t)(uintptr_t)p; }

// Broadcast a wave-uniform value into an SGPR via readfirstlane → scalar addressing, not a per-lane VGPR.
__device__ __forceinline__ uint32_t to_scalar(uint32_t v) { return __builtin_amdgcn_readfirstlane(v); }

// Immediate-offset ds_read2_b64: Offset0/Offset1 (8-B units) baked in; <0,16> fetches a d-tile pair in one op
// (offset1=16=128B=HeadDimTiles/2 ahead). d_pair rides the swizzled base, not the immediate; %c emits the bare const.
template <int Offset0, int Offset1> __device__ __forceinline__ u32x4_t ds_read2_b64_imm(uint32_t lds_base) {
    u32x4_t result;
    asm volatile("ds_read2_b64 %0, %1 offset0:%c2 offset1:%c3" : "=&v"(result) : "v"(lds_base), "i"(Offset0), "i"(Offset1));
    return result;
}

// ─── Wait-counts, barriers, scheduling ───────────────────────────────────────
// Pinned lgkmcnt wait — anchors the consumer so the compiler can't reorder above it. Count = LDS reads left in flight.
#define WAIT_LGKM_CASE(N)                                                                                                                                                          \
    if constexpr (Count == N)                                                                                                                                                      \
        asm volatile("s_waitcnt lgkmcnt(" #N ")" : "+v"(anchor)::"memory");
template <int Count, typename T> __device__ __forceinline__ void wait_lgkm(T &anchor) {
    static_assert(Count >= 0 && Count <= 7, "Count out of range");
    WAIT_LGKM_CASE(0)
    else WAIT_LGKM_CASE(1) else WAIT_LGKM_CASE(2) else WAIT_LGKM_CASE(3) else WAIT_LGKM_CASE(4) else WAIT_LGKM_CASE(5) else WAIT_LGKM_CASE(6) else WAIT_LGKM_CASE(7);
}
#undef WAIT_LGKM_CASE

__device__ __forceinline__ void wait_vmcnt0() { asm volatile("s_waitcnt vmcnt(0)" ::: "memory"); }
__device__ __forceinline__ void wait_lgkmcnt0() { __builtin_amdgcn_s_waitcnt(0x007F); }
__device__ __forceinline__ void lds_barrier() {
    wait_lgkmcnt0(); // lgkmcnt(0)
    __builtin_amdgcn_s_barrier();
}
// Issue-priority: the MFMA-heavy group raises prio to win issue arbitration over the memory/VALU group, then lowers.
// sched_fence = compiler scheduling barrier (pins order so hand-placed prefetches don't sink past it).
__device__ __forceinline__ void mfma_prio_hi() { asm volatile("s_setprio 3" ::: "memory"); }
__device__ __forceinline__ void mfma_prio_lo() { asm volatile("s_setprio 0" ::: "memory"); }
__device__ __forceinline__ void sched_fence() { __builtin_amdgcn_sched_barrier(0); }

// ─── HBM→LDS direct load ─────────────────────────────────────────────────────
// HBM→LDS direct dword load. Inline-asm `s_mov_b32 m0`: Clang's cselect around the builtin can sub m0=-1 when
// lds_dest is small, silently dropping writes.
__device__ __forceinline__ void
buffer_load_b32_to_lds(__amdgpu_buffer_rsrc_t buffer, uint32_t lds_dest, uint32_t voff, uint32_t soff) {
  asm volatile(
      "s_mov_b32 m0, %0\n\t"
      "buffer_load_dword %1, %2, %3 offen lds"
      :
      : "s"(lds_dest), "v"(voff), "s"(buffer), "s"(soff)
      : "m0", "memory");
}


// ─── Cross-lane reductions ───────────────────────────────────────────────────
// Cross-lane reduce over {l,l^16,l^32,l^48}: 4 quad-lanes sharing col span all 64 kvs. ds_swizzle 0x401F folds
// l^16; ds_bpermute folds l^32 (covers l^48). bperm_xor32: source-lane ((lane^32)<<2) computed INSIDE the asm from
// mbcnt each call (pure/remat) so no long-lived index VGPR SPILLS at the softmax peak (256-VGPR); lgkmcnt(0) baked
// in after bpermute. In asm: <<2 = lane→byte (4B/lane); XOR 0x80 = the "^32" peer (32 lanes·4B).
__device__ __forceinline__ int bperm_xor32(int data) {
  int idx, r;
  asm volatile(
      "v_mbcnt_lo_u32_b32 %0, -1, 0\n"
      "v_mbcnt_hi_u32_b32 %0, -1, %0\n"
      "v_lshlrev_b32_e32 %0, 2, %0\n"
      "v_xor_b32_e32 %0, 0x80, %0\n"
      "ds_bpermute_b32 %1, %0, %2\n"
      "s_waitcnt lgkmcnt(0)"
      : "=&v"(idx), "=&v"(r)
      : "v"(data)
      : "memory");
  return r;
}
__device__ __forceinline__ float cross_lane_max(float v) {
  const int peer16 = __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, v), 0x401F);
  const float partial = fmaxf(v, __builtin_bit_cast(float, peer16));
  const int peer32 = bperm_xor32(__builtin_bit_cast(int, partial));
  return fmaxf(partial, __builtin_bit_cast(float, peer32));
}
__device__ __forceinline__ float cross_lane_sum(float v) {
  const int peer16 = __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, v), 0x401F);
  const float partial = v + __builtin_bit_cast(float, peer16);
  const int peer32 = bperm_xor32(__builtin_bit_cast(int, partial));
  return partial + __builtin_bit_cast(float, peer32);
}

// ─── opus layouts + index maps ───────────────────────────────────────────────
template <int... S>
constexpr auto strides() {
  return opus::make_tuple(opus::number<S>{}...);
}
constexpr auto Shape2 = opus::make_tuple(opus::number<1>{}, opus::number<1>{});
constexpr auto Shape3 = opus::make_tuple(opus::number<1>{}, opus::number<1>{}, opus::number<1>{});

// Q / Out: row-major (seq, head_dim).
constexpr auto QOLayout = opus::make_layout(Shape2, strides<HeadDim, 1>());

// V_t (pre-transposed): (kv-iter, super-chunk, lane). bf16x8 / lane.
constexpr int VtChunks = 16;                           // super-chunks / kv-block
constexpr int VtBlockElems = KvBlockRows * HeadDim;    // 8192 elems / kv-block
constexpr int VtChunkElems = VtBlockElems / VtChunks;  // 512  elems / super-chunk
constexpr int VtLaneElems = 8;                         // bf16x8 / lane
constexpr auto VtLayout = opus::make_layout(Shape3, strides<VtBlockElems, VtChunkElems, VtLaneElems>());

// K in DWORDS (2 bf16), row-major (kv_row,dword), 64 dw/row. LDS dest: same stride 64 but XOR-swizzled
// (direct_lds_k/qk_swz_base) to break the 32-way qk-read bank conflict (stride 64≡0 mod 32) without padding (pad is
// slower).
constexpr auto KLdsDword = opus::make_layout(Shape2, strides<KLdsStrideDwords, 1>());

// HBM (seq, head-dim/dword) → Q/K/Out offset layout by input layout. BHSD: per-head contiguous → static stride
// Stride (byte-identical). BSHD: heads interleaved → dynamic stride heads·Stride. Unused branch folds away.
template <bool kBSHD, int Stride>
__device__ __forceinline__ auto seq_layout(int heads) {
  if constexpr (kBSHD)
    return opus::make_layout(Shape2, opus::make_tuple(heads * Stride, opus::number<1>{}));
  else
    return opus::make_layout(Shape2, opus::make_tuple(opus::number<Stride>{}, opus::number<1>{}));
}

// qg2 LDS park: entire q-group 2 (all HeadDimTiles d-tiles) staged in LDS, addressed in DWORDS by
// (wave, d-tile, lane); each lane holds a bf16x4 (2 dwords) → lane stride 2.
constexpr int Q2HalfTiles = HeadDimTiles / 2;  // 4 = #d-tiles per half; also the lo→hi stride (d_pair → d_pair+4)
constexpr int Q2LdsDwords = NumWaves * HeadDimTiles * WaveSize * 2;  // 8192 dwords = whole qg2 (32 KB)
constexpr auto Q2Lds = opus::make_layout(Shape3, strides<HeadDimTiles * WaveSize * 2, WaveSize * 2, 2>());
// LITE skip-vote ring: live window {step,step-1} → depth 2. Reuses the 4 KB lds_q2 slot freed by evicting qg2's last
// d-tile to VGPR.
constexpr int VerdictRing = 2;

// ─── Grid + MFMA thread→fragment index maps ─────────────────────────────────
// Same rule as the memory layouts (offset = Σ coordᵢ·strideᵢ) but yield LOGICAL indices (q-tile/kv-row/frag elem), not
// bytes.

// (block_x, wave, q-group) → absolute q-tile index (WG owns QTilesPerWG, wave QTilesPerWave, q-group 1).
constexpr auto QTileGrid = opus::make_layout(Shape3, strides<QTilesPerWG, QTilesPerWave, 1>());

// (load-wave, local row) → kv-row owned: LoadWaves × KRowsPerLoadWave tile the K block.
constexpr auto KLoadRow = opus::make_layout(Shape2, strides<KRowsPerLoadWave, 1>());

// (load-wave, i) → V_t super-chunk to L1-prefetch (VtChunks chunks / LoadWaves).
constexpr auto VtPrefetch = opus::make_layout(Shape2, strides<VtChunks / LoadWaves, 1>());

// MFMA 16x16x16 A/B fragment: lane=(col=lane&15, quad=lane>>4); 16-wide contraction tiled by 4 quad-lanes, QuadSpan
// elems each.
constexpr int QuadSpan = MfmaK / 4;  // 4 bf16 contraction elems / quad-lane
// (d-tile, quad) → head-dim element, for Q / Out (bf16 elements).
constexpr auto HdFrag = opus::make_layout(Shape2, strides<MfmaK, QuadSpan>());

// Head-first chiplet swizzle: chunk_size=blocks_per_head co-locates one (B,H)'s q-blocks on one XCD's L2 → K+V stay
// resident.
constexpr int NumXcds = 8;
__device__ __forceinline__ int chiplet_swizzle(int wg_id, int num_wgs, int chunk_size) {
  const int block = NumXcds * chunk_size;
  const int limit = (num_wgs / block) * block;
  if (wg_id >= limit) return wg_id;
  const int xcd = wg_id % NumXcds;
  const int local_id = wg_id / NumXcds;
  const int chunk_idx = local_id / chunk_size;
  const int pos_in_chunk = local_id % chunk_size;
  return chunk_idx * block + xcd * chunk_size + pos_in_chunk;
}

// ─── Dense tail KV-split scratch (dense only) ────────────────────────────────
// 384-row q-blocks run in rounds of NumCUs; a fractional last round strands CUs. Recover it by splitting each
// stranded q-block's kv work G ways (split_ways): part p walks every G-th kv block from offset p, writing an fp32
// (o,m,l) partial; merge_parts recombines a q-block's G partials via online-softmax into bf16.
constexpr int NumCUs = 304;           // MI300X compute units (= one occupancy-1 CTA each)
constexpr int MaxParts = 2 * NumCUs;  // upper bound on total parts (the planner caps the split at ≤2 CU rounds)
constexpr int PartRows = QTilesPerWG * QTileRows;  // 384 q-rows owned by one part (== one q-block)
constexpr int PartOFloats = PartRows * HeadDim;    // fp32 `o` values per part (384·128)
constexpr int PartMlFloats = PartRows * 2;         // fp32 (m,l) pair per row, per part (384·2)
// One flat float buffer: `o` region then `(m,l)` region, each addressed by a compile-time-strided opus layout.
constexpr auto PartO = opus::make_layout(Shape3, strides<PartOFloats, HeadDim, 1>());  // (part, row, hd) → o
constexpr auto PartMl = opus::make_layout(Shape2, strides<PartMlFloats, 2>());         // (part, row) → m; +1 → l
struct TailScratch {
  float* o;   // [MaxParts][PartRows][HeadDim] fp32 output partials, indexed by PartO(part, row, hd)
  float* ml;  // [MaxParts][PartRows][2]       fp32 (row max, row sum), indexed by PartMl(part, row){+1}
  __device__ __host__ explicit TailScratch(float* ws) : o(ws), ml(ws + (long long)MaxParts * PartOFloats) {}
};
// Online-softmax update for ONE q-group (per-q-group scoping keeps kernel <256-VGPR; inlining all spills ~110). Folds
// block into running (max,sum,o_acc): skip-rescale gate, exp2+NaN-safe pack, row_sum; cross-lane sum to epilogue.
__device__ __forceinline__ void softmax_update(fp32x4_t scores[], fp32x4_t oreg[], bf16x4_t probs[], float &row_m, float &row_l, float scale_log2e, float bmax_local) {
    const float new_m_local = bmax_local * scale_log2e;
    const bool first_block = row_m == -FLT_MAX;
    const bool lane_ok = (new_m_local - row_m) < RescaleThreshold;
    const bool all_ok = wave_all(lane_ok) && !first_block;

    float neg_max;
    float rescale = 1.f;
    if (__builtin_expect(all_ok, 1)) {
        neg_max = -row_m;
    } else {
        const float bmax = cross_lane_max(bmax_local);
        const float old_m = row_m;
        const float new_m = fmaxf(old_m, bmax * scale_log2e);
        if (new_m != old_m) {
            rescale = first_block ? 0.f : fast_exp2(old_m - new_m);
            if (!first_block) {
                // o_acc writeback fence: "+v"(oreg)+"memory" orders rescale after pv's asm MFMA writes (LLVM can't see those
                // defs); same-wave MFMA→VALU HW interlock covers writeback.
                asm volatile("" : "+v"(oreg[0]), "+v"(oreg[1]), "+v"(oreg[2]), "+v"(oreg[3]), "+v"(oreg[4]), "+v"(oreg[5]), "+v"(oreg[6]), "+v"(oreg[7])::"memory");
                #pragma unroll
                for (int t = 0; t < HeadDimTiles; ++t) {
                    asm volatile("v_mul_f32_e32 %0, %1, %0" : "+v"(oreg[t][0]) : "v"(rescale));
                    asm volatile("v_mul_f32_e32 %0, %1, %0" : "+v"(oreg[t][1]) : "v"(rescale));
                    asm volatile("v_mul_f32_e32 %0, %1, %0" : "+v"(oreg[t][2]) : "v"(rescale));
                    asm volatile("v_mul_f32_e32 %0, %1, %0" : "+v"(oreg[t][3]) : "v"(rescale));
                }
            }
        }
        row_m = new_m;
        neg_max = -new_m;
    }

    float sub_sum[KvSubtiles];
    #pragma unroll
    for (int kv_sub = 0; kv_sub < KvSubtiles; ++kv_sub) {
        const float p0 = fast_exp2(fmaf(scores[kv_sub][0], scale_log2e, neg_max));
        const float p1 = fast_exp2(fmaf(scores[kv_sub][1], scale_log2e, neg_max));
        const float p2 = fast_exp2(fmaf(scores[kv_sub][2], scale_log2e, neg_max));
        const float p3 = fast_exp2(fmaf(scores[kv_sub][3], scale_log2e, neg_max));
        sub_sum[kv_sub] = (p0 + p1) + (p2 + p3);
        // No NaN guard (hot path): exp2 → [0,1] or NaN (never Inf); NaN packs to bf16 NaN → PV MFMA → fp32 NaN → epilogue
        // pack canonicalizes to 0x7FFF. Guarding = cmp+cndmask/elem for no effect.
        probs[kv_sub] = pack4_bf16</*MaybeNaN=*/false>(p0, p1, p2, p3);
    }
    const float block_sum = vadd_f32(vadd_f32(sub_sum[0], sub_sum[1]), vadd_f32(sub_sum[2], sub_sum[3]));
    row_l = vfma_f32(row_l, rescale, block_sum);
}

template <bool kBSHD, bool isLite = false, bool Split = false>
__global__ __launch_bounds__(BlockSize, 1) void attention_forward(const bf16_t *__restrict__ Q, const bf16_t *__restrict__ K,
                                                                  const bf16_t *__restrict__ V, // points to V_t (pre-transposed)
                                                                  bf16_t *__restrict__ Out, int seq_len, int seq_len_kv,
                                                                  float scale_log2e, // seq_len = Q rows; seq_len_kv = K/V rows (cross-attn)
                                                                  int heads,         // #heads — only needed for BSHD addressing (kBSHD)
                                                                  // ── LiteAttention skip (isLite only; dense passes the defaults) ──
                                                                  const int16_t *__restrict__ read_list = nullptr,    // this phase's RLE keep-list (one row per CTA)
                                                                  int16_t *__restrict__ write_list = nullptr,         // RLE keep-list emitted for the next timestep
                                                                  const int16_t *__restrict__ must_do_list = nullptr, // global force-compute ranges [len,s0,e0,…] (nullable)
                                                                  float thr = 0.f,
                                                                  int phase = 0, // skip threshold (log2) + reverse-list phase
                                                                  // ── Tail KV-split ──
                                                                  float *__restrict__ part_buf = nullptr, // Split: the TailScratch workspace
                                                                  int grid_nqb = 0,                       // real #q-blocks when the grid is 1-D (prefix/split);
                                                                                                          //   0 ⇒ 2-D full grid (uses gridDim.x)
                                                                  int split_ways = 0,                     // Split: G = strided parts per q-block (= the kv stride)
                                                                  // ── LiteAttention LPT work-balance reorder (lite full grid only) ──
                                                                  const int *__restrict__ lpt_order = nullptr // lpt_order[grid slot] = work-group to run here (longest-first)
);
// ===== V TRANSPOSE: V → V_t pre-transpose for the PV mfma

// ─── V transpose for 16x16x16 mfma ─────────────────────────────────────────────
// Pre-packs V[seq,128] into V_t for pv_phase's one contiguous load_u32x4/lane. Phase 1: V block HBM→LDS coalesced;
// phase 2: re-read in fragment order → V_t (VtLayout). super-chunk = d_tile*2 + kv_sub_pair = 64 A-frags; lane l's 16 B
// = 2 A-operands. l=frag_lane 0..63, i=0..3 = bf16 in operand:
//   bytes [l*16 .. l*16+7]   = V[k=kv_sub0*16 + (l/16)*4 + i, hd=d_tile*16 + l%16]
//   bytes [l*16+8 .. l*16+15] = V[k=kv_sub1*16 + (l/16)*4 + i, hd=d_tile*16 + l%16]

constexpr int XposeThreadsPerWG = 256;
constexpr auto VStageWr = opus::make_layout(Shape2, strides<HeadDim / 2, 4>());               // (row, octet)→dword
constexpr auto VtChunkLane = opus::make_layout(Shape2, strides<VtChunkElems, VtLaneElems>()); // (super-chunk, lane)
template <bool kBSHD>
__global__ __launch_bounds__(XposeThreadsPerWG, 4) void v_transpose_kernel_16x16(const bf16_t *__restrict__ V, bf16_t *__restrict__ V_t, int seq_len_per_head,
                                                                                 int heads // #heads — only for BSHD V addressing (kBSHD)
) {
    const int num_kv_blocks = (seq_len_per_head + KvBlockRows - 1) / KvBlockRows;
    const int head = blockIdx.x / num_kv_blocks;
    const int kv_iter = blockIdx.x % num_kv_blocks;
    const int tid = threadIdx.x;
    // head = bh index over B·H; BSHD splits it into b=head/heads, h=head%heads.
    long long src_block_base;
    if constexpr (kBSHD) {
        const long long b = head / heads, h = head % heads;
        src_block_base = b * (long long)seq_len_per_head * heads * HeadDim + h * HeadDim + (long long)kv_iter * KvBlockRows * heads * HeadDim; // unpadded BSHD source base
    } else {
        src_block_base = (long long)head * seq_len_per_head * HeadDim + (long long)kv_iter * VtBlockElems; // unpadded BHSD source base
    }
    const long long dst_block_base = (long long)head * num_kv_blocks * VtBlockElems + (long long)kv_iter * VtBlockElems;

    __shared__ uint32_t stage_dw[KvBlockRows * (HeadDim / 2)]; // 64 rows × 64 dw = 16 KB
    const auto v_buffer = make_buffer(V);

    // ── Phase 1: V[block] HBM → LDS, coalesced; rows ≥ seq_len_per_head zero (padded tail). Shadow QOLayout with the
    // source-V seq-row stride (phase-1 only; phase 2 reads the contiguous LDS stage via global QOLayout).
    {
        const auto QOLayout = seq_layout<kBSHD, HeadDim>(heads);
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int elem_idx = k * XposeThreadsPerWG + tid;
            const int row = elem_idx >> 4;   // 0..63 kv-row in this block
            const int octet = elem_idx & 15; // 0..15: which b128 load-unit (8 bf16 = 16 B) along the 128 head-dim
            const bool row_valid = (kv_iter * KvBlockRows + row) < seq_len_per_head;
            u32x4_t v_oct = row_valid ? __builtin_amdgcn_raw_buffer_load_b128(v_buffer, (uint32_t)((src_block_base + QOLayout(row, octet * 8)) * 2), 0, 0) : u32x4_t{0, 0, 0, 0};
            const int dst_dw = VStageWr(row, octet); // dword offset in stage_dw
            stage_dw[dst_dw + 0] = v_oct[0];
            stage_dw[dst_dw + 1] = v_oct[1];
            stage_dw[dst_dw + 2] = v_oct[2];
            stage_dw[dst_dw + 3] = v_oct[3];
        }
    }

    __syncthreads();

    // staged tile is bf16, row-major (kv_row, head_dim) — same shape as Q (QOLayout).
    const bf16_t *stage = reinterpret_cast<const bf16_t *>(stage_dw);
    const auto vt_buffer = make_buffer(V_t);

    // ── Phase 2: gather each (super-chunk, lane)'s two A-fragments from LDS → V_t. ──
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int frag_idx = tid * 4 + k;    // 0..1023 = VtChunks(16) super-chunks × 64 lanes (256 threads × 4)
        const int chunk = frag_idx >> 6;     // super-chunk 0..15
        const int frag_lane = frag_idx & 63; // lane 0..63
        const int d_tile = chunk >> 1;       // hd-tile 0..7
        const int kv_sub_pair = chunk & 1;   // 0..1
        const int kv_sub0 = kv_sub_pair * 2;
        const int kv_sub1 = kv_sub0 + 1;
        const int hd = d_tile * 16 + (frag_lane & 15); // head-dim element 0..127
        const int kv_row0 = (frag_lane >> 4) * 4;      // hi 2 bits of lane pick a row-quad: owns kv-rows kv_row0..+3

        bf16_t packed[8];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            packed[i] = stage[QOLayout(kv_sub0 * 16 + kv_row0 + i, hd)];
            packed[4 + i] = stage[QOLayout(kv_sub1 * 16 + kv_row0 + i, hd)];
        }
        u32x4_t packed_v;
        __builtin_memcpy(&packed_v, packed, sizeof(packed));
        __builtin_amdgcn_raw_buffer_store_b128(packed_v, vt_buffer, (uint32_t)((dst_block_base + VtChunkLane(chunk, frag_lane)) * 2), 0, 0);
    }
}

// ===== DISPATCH: tail-part merge kernel + planners + launch glue

// ─── Tail-part merge kernel ──────────────────────────────────────────────────
constexpr int MergeThreads = 256;                               // threads per merge CTA
constexpr int MergeWGs = PartRows * HeadDim / 4 / MergeThreads; // 48 CTAs per q-block's 384×128 output

// Per-(B,H) base elem offset + SRD byte span of [seq,(heads,)HeadDim] output for head-row `bh`. BHSD: heads contiguous;
// BSHD: interleaved → ×heads span, start at h·HeadDim.
template <bool kBSHD> __device__ __forceinline__ long long head_base_offset(int bh, int seq, int heads) {
    if constexpr (kBSHD)
        return (long long)(bh / heads) * seq * heads * HeadDim + (long long)(bh % heads) * HeadDim;
    else
        return (long long)bh * seq * HeadDim;
}
template <bool kBSHD> __device__ __forceinline__ uint32_t head_span_bytes(int seq, int heads) {
    return (uint32_t)((long long)seq * (kBSHD ? heads : 1) * HeadDim * sizeof(bf16_t));
}



// Host-uploaded schedule (static single-stream buffer) read by split+merge kernels: [0,parts)=per-part
// (q_block,bh,_,_); [MaxParts,+tail)=per-q-block (q_block,bh,part_base,n_parts=G).
__device__ int4 tail_sched[2 * MaxParts];
// ===== FORWARD KERNEL: online softmax + attention_forward (dense / lite / split)



// Recombine a tail q-block's `n_parts`(=G) strided KV partials. Each thread owns ONE output cell = 4 head-dims of one
// row. Online-softmax merge: run_max = maxₚ mₚ, weight wₚ = exp2(mₚ−run_max), out = (Σₚ wₚ·oₚ)/(Σₚ wₚ·lₚ), bf16-packed.
template <bool kBSHD> __global__ __launch_bounds__(MergeThreads, 4) void merge_parts(bf16_t *__restrict__ Out, int seq_len, int heads, const float *__restrict__ part_buf) {
    // tail_sched (see plan_tail_split): [0,MaxParts) per-part records, [MaxParts,+tail) per-q-block meta = {q_block=.x,
    // bh=.y, part_base=.z, n_parts=.w}.
    const int4 meta = tail_sched[MaxParts + blockIdx.x / MergeWGs];
    const int part_base = meta.z, n_parts = meta.w;
    // This thread's output cell: (row, hd), hd 4-aligned.
    const int cell = ((blockIdx.x % MergeWGs) * MergeThreads + threadIdx.x) * 4;
    const int row = cell / HeadDim, hd = cell % HeadDim;
    const TailScratch parts(const_cast<float *>(part_buf));

    // Pass 1: run_max = max of the parts' softmax maxima (each m is a lower bound on the true max).
    float run_max = -FLT_MAX;
    for (int p = 0; p < n_parts; ++p)
        run_max = fmaxf(run_max, parts.ml[PartMl(part_base + p, row)]);
    // Pass 2: weighted numerators (out0..out3) + shared denom l_sum.
    float out0 = 0, out1 = 0, out2 = 0, out3 = 0, l_sum = 0;
    for (int p = 0; p < n_parts; ++p) {
        const float *ml = &parts.ml[PartMl(part_base + p, row)];  // ml[0] = m_p, ml[1] = l_p
        const float *o = &parts.o[PartO(part_base + p, row, hd)]; // o_p[hd .. hd+3]
        const float weight = fast_exp2(ml[0] - run_max);
        out0 = fmaf(weight, o[0], out0);
        out1 = fmaf(weight, o[1], out1);
        out2 = fmaf(weight, o[2], out2);
        l_sum = fmaf(weight, ml[1], l_sum); // denom interleaved between out2/out3 deliberately:
        out3 = fmaf(weight, o[3], out3);    // splits the 4-wide FMA dependency chain for the scheduler.
    }
    const float inv_l = (l_sum == 0.0f) ? 0.0f : fast_rcp(l_sum);

    // Scatter normalized bf16x4 to Out at (q-block row, hd).
    const auto OutLayout = seq_layout<kBSHD, HeadDim>(heads);
    const auto out_buffer = make_buffer(Out + head_base_offset<kBSHD>(meta.y, seq_len, heads), head_span_bytes<kBSHD>(seq_len, heads));
    store_bf16x4(out_buffer, OutLayout(meta.x * PartRows + row, hd) * 2, pack4_bf16</*MaybeNaN=*/true>(out0 * inv_l, out1 * inv_l, out2 * inv_l, out3 * inv_l));
}

// ─── Planners ────────────────────────────────────────────────────────────────
// ─── Tail KV-split planner (dense) ───────────────────────────────────────────
// Recover the stranded fractional CU-round by KV-splitting it, uploading a schedule if it pays off. "No split"
// (tail==0, split_ways==1: whole grid runs as ordinary prefix) when not worth the merge; parts<0 signals a
// schedule-upload failure.
struct TailPlan {
    int prefix_wgs; // CTAs run as the ordinary full-stream prefix (wgs − tail; 0 ⇒ none)
    int tail;       // stranded q-blocks recovered by the split (0 ⇒ no split at all)
    int split_ways; // G: number of strided parts per stranded q-block
    int parts;      // total split parts = tail·G (<0 ⇒ upload error)
};
constexpr int MinKvBlocksToSplit = 48; // below this, the merge can't amortize — don't split
constexpr int MinBlocksPerPart = 4;    // floor on kv blocks/part — finer splits let fixed cost dominate
constexpr int MergeCostBlocks = 3;     // merge kernel's modelled cost, in kv-block-time units

// ─── LiteAttention LPT (longest-processing-time) work-balance order ───────────
// Lite is work-bound: each q-block keeps a different # of K-blocks; the occ-1 grid's dynamic dispatch balances the
// bulk, leaving only ramp-down loss when a heavy q-block dispatches late. LPT runs heaviest q-blocks first → makespan ≈
// total_work/NumCUs. We only choose the ORDER (counting sort by kept-count, desc); HW dispatch IS the assignment.
// lpt_order[slot] = WG the CTA at grid slot `slot` runs. Each WG owns its own rows → bit-identical, no split/merge.
constexpr int kWorkBuckets = 256; // kept-count buckets for the counting sort (coarse order is fine)
__global__ __launch_bounds__(256, 1) void plan_lite_order(const int16_t *__restrict__ read_list, int *__restrict__ lpt_order, int num_q_blocks, int num_kv_blocks, int num_bh,
                                                          int phase) {
    const int num_wgs = num_q_blocks * num_bh;
    const int lstep = phase ? 1 : -1;
    __shared__ int bucket_count[kWorkBuckets]; // # work-groups whose work falls in each bucket
    __shared__ int bucket_slot[kWorkBuckets];  // running next free output slot per bucket
    for (int b = threadIdx.x; b < kWorkBuckets; b += blockDim.x)
        bucket_count[b] = 0;
    __syncthreads();

    // Count kept K-blocks from the WG's skip-list RLE row [len, hi0,lo0, …] (len int16s of (hi,lo) pairs), bucketize by
    // count. +lstep rebases endpoints into phase; kept/pair=|hi-lo|. e.g. (lstep=+1) row=[4,9,5,20,18] → 4+2 = kept 6.
    auto work_bucket = [&](int wg) -> int {
        const int q_block = wg % num_q_blocks, bh = wg / num_q_blocks;
        const long long row = ((long long)bh * num_q_blocks + q_block) * (num_kv_blocks + 2);
        const int len = (int)read_list[row];
        int kept = 0;
        for (int i = len; i >= 2; i -= 2) {
            const int lo = (int)read_list[row + i] + lstep;
            const int hi = (int)read_list[row + i - 1] + lstep;
            kept += (lstep > 0) ? (hi - lo) : (lo - hi);
        }
        const int bucket = kept * kWorkBuckets / (num_kv_blocks + 1);
        return bucket < kWorkBuckets ? bucket : kWorkBuckets - 1; // clamp the top bucket
    };

    for (int wg = threadIdx.x; wg < num_wgs; wg += blockDim.x)
        atomicAdd(&bucket_count[work_bucket(wg)], 1);
    __syncthreads();
    // Exclusive scan HIGH→LOW so the heaviest buckets get the earliest grid slots (longest-first).
    if (threadIdx.x == 0) {
        int slot = 0;
        for (int b = kWorkBuckets - 1; b >= 0; --b) {
            bucket_slot[b] = slot;
            slot += bucket_count[b];
        }
    }
    __syncthreads();
    for (int wg = threadIdx.x; wg < num_wgs; wg += blockDim.x)
        lpt_order[atomicAdd(&bucket_slot[work_bucket(wg)], 1)] = wg;
}

TailPlan plan_tail_split(int nqb, int num_bh, int nkb, bool enable, hipStream_t stream) {
    const int wgs = nqb * num_bh;
    const TailPlan no_split{wgs, 0, 1, 0};
    const int tail = enable ? (wgs % NumCUs) : 0; // CTAs left over after full CU-rounds = the stranded round
    // Single ordinary launch (no split) when: grid tiles evenly, last round ≥95% full (too little idle), or kv too short
    // for the merge to pay off.
    const bool grid_tiles_evenly = !tail;
    const bool last_round_nearly_full = tail > NumCUs * 19 / 20;
    const bool kv_too_short = nkb < MinKvBlocksToSplit;
    if (grid_tiles_evenly || last_round_nearly_full || kv_too_short)
        return no_split;

    // Choose G minimizing span = ceil(tail·G/NumCUs) rounds × ceil(nkb/G) blocks/part + merge cost (G=1 = whole nkb).
    // e.g. tail=40,nkb=192,NumCUs=304: G=1 → 192+3=195; G=4 → 160 parts (1 round)×48 = 48+3=51 (wins).
    int G = 1;
    long long best_span = (long long)nkb + MergeCostBlocks;
    for (int g = 2; g <= MaxParts / tail && nkb / g >= MinBlocksPerPart; ++g) {
        const long long cu_rounds = (tail * g + NumCUs - 1) / NumCUs; // ceil(tail·g / NumCUs)
        const long long blocks_per_part = (nkb + g - 1) / g;          // ceil(nkb / g)
        const long long span = cu_rounds * blocks_per_part + MergeCostBlocks;
        if (span < best_span) {
            best_span = span;
            G = g;
        }
    }
    if (G == 1)
        return no_split; // no split factor beats running the full stranded round

    // Per stranded q-block: G strided parts + one merge-meta entry. NOTE: static `sched` reused across calls —
    // correctness relies on the single-stream contract (stream-ordered H2D before the next plan overwrites it;
    // multi-stream is UB).
    static int4 sched[2 * MaxParts];
    int parts = 0;
    for (int t = 0; t < tail; ++t) {
        const int wg = (wgs - tail) + t; // stranded CTA index, in the prefix's bh-major order
        const int q_block = wg % nqb, bh = wg / nqb;
        sched[MaxParts + t] = int4{q_block, bh, parts, G}; // per-q-block: (q_block, bh, part_base, n_parts)
        for (int i = 0; i < G; ++i)
            sched[parts + i] = int4{q_block, bh, 0, 0}; // per-part:    (q_block, bh, _, _)
        parts += G;
    }
    if (hipMemcpyToSymbolAsync(HIP_SYMBOL(tail_sched), sched, sizeof(int4) * (MaxParts + tail), 0, hipMemcpyHostToDevice, stream) != hipSuccess)
        return TailPlan{wgs, 0, 1, -1};
    return TailPlan{wgs - tail, tail, G, parts};
}

// ─── Launch glue ─────────────────────────────────────────────────────────────
// Single launch site for the forward kernel: resolves runtime `bshd` to the kBSHD template + forwards defaulted args.
template <bool isLite, bool Split>
void fwd_launch(bool bshd, dim3 grid, hipStream_t stream, const void *q, const void *k, const void *v, void *out, int seq_len, int seq_len_kv, float scale_log2e, int heads,
                const void *read_list = nullptr, void *write_list = nullptr, const void *must_do_list = nullptr, float thr = 0.f, int phase = 0, float *part_buf = nullptr,
                int grid_nqb = 0, int split_ways = 0, const int *lpt_order = nullptr) {
#define FWD_ARGS                                                                                                                                                                   \
    (const bf16_t *)q, (const bf16_t *)k, (const bf16_t *)v, (bf16_t *)out, seq_len, seq_len_kv, scale_log2e, heads, (const int16_t *)read_list, (int16_t *)write_list,            \
        (const int16_t *)must_do_list, thr, phase, part_buf, grid_nqb, split_ways, lpt_order
    if (bshd)
        attention_forward<true, isLite, Split><<<grid, dim3(BlockSize), 0, stream>>>(FWD_ARGS);
    else
        attention_forward<false, isLite, Split><<<grid, dim3(BlockSize), 0, stream>>>(FWD_ARGS);
#undef FWD_ARGS
}

// Dense tail KV-split: prefix (if non-empty) → split parts → merge, all stream-ordered.
void launch_tail_split(bool bshd, const void *q, const void *k, const void *v, void *out, int seq_len, int seq_len_kv, float scale_log2e, int heads, int nqb, void *workspace,
                       const TailPlan &tp, hipStream_t stream) {
    // Prefix: full-stream waves as one 1-D bh-major launch (grid_nqb=nqb strips the tail); empty when the whole grid is
    // the stranded round.
    if (tp.prefix_wgs)
        fwd_launch</*isLite=*/false, /*Split=*/false>(bshd, dim3(tp.prefix_wgs), stream, q, k, v, out, seq_len, seq_len_kv, scale_log2e, heads, nullptr, nullptr, nullptr, 0.f, 0,
                                                      /*part_buf=*/nullptr,
                                                      /*grid_nqb=*/nqb);
    // Split: one CTA per part, each walking every G-th kv block of its q-block into an fp32 partial.
    fwd_launch</*isLite=*/false, /*Split=*/true>(bshd, dim3(tp.parts), stream, q, k, v, out, seq_len, seq_len_kv, scale_log2e, heads, nullptr, nullptr, nullptr, 0.f, 0,
                                                 /*part_buf=*/(float *)workspace,
                                                 /*grid_nqb=*/nqb,
                                                 /*split_ways=*/tp.split_ways);
    // Merge: recombine each stranded q-block's parts into the final bf16 output.
    if (bshd)
        merge_parts<true><<<dim3(tp.tail * MergeWGs), dim3(MergeThreads), 0, stream>>>((bf16_t *)out, seq_len, heads, (const float *)workspace);
    else
        merge_parts<false><<<dim3(tp.tail * MergeWGs), dim3(MergeThreads), 0, stream>>>((bf16_t *)out, seq_len, heads, (const float *)workspace);
}

// Lite full-grid launch with LPT reordering: grid slot i processes lpt_order[i]. Pure grid-slot→WG reorder →
// bit-identical, tighter makespan. `order_buf` = num_wgs ints (the permutation).
void launch_lite_ordered(bool bshd, const void *q, const void *k, const void *v, void *out, int seq_len, int seq_len_kv, float scale_log2e, int heads, int nqb, int bh, int nkb,
                         const void *read_list, void *write_list, const void *must_do_list, float threshold, int phase, void *order_buf, hipStream_t stream) {
    int *lpt_order = (int *)order_buf;
    plan_lite_order<<<1, dim3(256), 0, stream>>>((const int16_t *)read_list, lpt_order, nqb, nkb, bh, phase);
    fwd_launch</*isLite=*/true, /*Split=*/false>(bshd, dim3(nqb * bh), stream, q, k, v, out, seq_len, seq_len_kv, scale_log2e, heads, read_list, write_list, must_do_list,
                                                 threshold, phase,
                                                 /*part_buf=*/nullptr,
                                                 /*grid_nqb=*/nqb,
                                                 /*split_ways=*/0, lpt_order);
}


} // namespace mmai_attention