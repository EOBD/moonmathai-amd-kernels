// Forward attention for AMD CDNA3 (MI300X / gfx942), bf16, FA2-style online softmax.
// 16x16x16 bf16 MFMA. 8-wave warp-spec CTA, two groups of 4 waves; per K-tile each
// iter runs as two phases:
//
//                    Group A                      Group B
//   ───────  ────────────────────────  ────────────────────────────
//   Phase 1  pv[N-1] + qk[N]           K[N+1] HBM→LDS + softmax[N-1]
//   Phase 2  softmax[N] + V_t prefetch  pv[N-1] + qk[N]
//
// One group keeps the MFMA engine active while the other does VALU+memory. L1V variant: V is
// pre-transposed offline (v_transpose_kernel_16x16) into V_t for direct load_u32x4 in pv_phase, no LDS staging.
//
// MFMA layout: lane col=lane&15, quad=lane>>4 (4 quads/fp32x4 C-frag); wave owns QTilesPerWave=3
// q-tiles of 16 rows; qk reads K in d-tile pairs {d_pair,d_pair+4}; pv consumes V_t super-chunks.
//
// One TU, top to bottom. Round modes built by attention_{rtna,rtne,rtz}.hip shims (#define launcher, #include this).

#include "attention_kernel.h"
using namespace mmai_attention;
namespace mmai_attention {

template <bool kBSHD, bool isLite, bool Split>
__global__ __launch_bounds__(BlockSize, 1) void attention_forward(const bf16_t *__restrict__ Q, const bf16_t *__restrict__ K,
                                                                  const bf16_t *__restrict__ V, // points to V_t (pre-transposed)
                                                                  bf16_t *__restrict__ Out, int seq_len, int seq_len_kv,
                                                                  float scale_log2e, // seq_len = Q rows; seq_len_kv = K/V rows (cross-attn)
                                                                  int heads,         // #heads — only needed for BSHD addressing (kBSHD)
                                                                  // ── LiteAttention skip (isLite only; dense passes the defaults) ──
                                                                  const int16_t *__restrict__ read_list,    // this phase's RLE keep-list (one row per CTA)
                                                                  int16_t *__restrict__ write_list,         // RLE keep-list emitted for the next timestep
                                                                  const int16_t *__restrict__ must_do_list, // global force-compute ranges [len,s0,e0,…] (nullable)
                                                                  float thr,
                                                                  int phase, // skip threshold (log2) + reverse-list phase
                                                                  // ── Tail KV-split ──
                                                                  float *__restrict__ part_buf, // Split: the TailScratch workspace
                                                                  int grid_nqb,                 // real #q-blocks when the grid is 1-D (prefix/split);
                                                                                                //   0 ⇒ 2-D full grid (uses gridDim.x)
                                                                  int split_ways,               // Split: G = strided parts per q-block (= the kv stride)
                                                                  // ── LiteAttention LPT work-balance reorder (lite full grid only) ──
                                                                  const int *__restrict__ lpt_order // lpt_order[grid slot] = work-group to run here (longest-first)
) {
    static_assert(!(isLite && Split), "lite has no tail KV-split; Split is dense-only");
    // ROADMAP (one CTA = one 384-row q-block of one (batch,head)):
    //   1. Thread/lane setup + per-(B,H) HBM base offsets for Q/K/V_t/Out.
    //   2. Prologue — Q→VGPR/LDS, first K-block→LDS, zero accumulators.
    //   3. Main loop over K-blocks — 8-wave warp-spec: 2 groups swap MFMA-vs-memory across a mid-iter barrier (qk/pv
    //   overlap softmax+K/V loads).
    //   4. Epilogue — finish last block; Split: raw fp32 partials to tail scratch, else normalize by denom + store bf16
    //   to Out.
    // Dense/lite(skip)/split share this body; `if constexpr` selects per template.
    //
    // MFMA 16x16x16 lane split: col = q-col/kv-row in tile, quad = which 4 of 64. 8 waves → 2 groups of 4 for warp-spec.
    const int tid = threadIdx.x;
    const int wave_id = tid >> 6;
    const int lane = tid & 63;
    const int col = lane & 15;  // q-col / kv-row within a 16-wide tile
    const int quad = lane >> 4; // which group of 4 lanes (0..3)
    const bool group_a = (wave_id < LoadWaves);
    const int wave_in_group = group_a ? wave_id : (wave_id - LoadWaves);

    // K-LDS XOR swizzle (stride-64, bank-conflict-free), matches direct_lds_k's HBM-source swizzle. Read-offset fields
    // are bit-disjoint (OR≡add): col[8-11] | kv_sub[12-13] | k_buf[14] | (quad^col_lo)[3-4] | (d_pair^col_hi)[5-6];
    // d-tile-high rides offset1=16.
    const uint32_t col_lo = (uint32_t)(col & 3);
    const uint32_t col_hi = (uint32_t)(col >> 2);
    // worked example col=5,quad=2: col_lo=1 → (5<<8) | ((2^1)<<3) = 0x500 | 0x18 = 0x518.
    const uint32_t qk_swz_base = (uint32_t)(((unsigned)col << 8) | (((unsigned)quad ^ col_lo) << 3));

    // Grid: x = #q-blocks (384-row = ceil(seq_len/384)), y = batch·heads. One CTA per (q-block, bh).
    const int num_q_blocks = gridDim.x;
    const int num_bh = gridDim.y;
    const int wg_id_raw = blockIdx.y * num_q_blocks + blockIdx.x;
    const int num_wgs = num_q_blocks * num_bh;
    // Block selection. Split: (q_block,bh) from tail_sched. 2-D full grid (grid_nqb==0): lite uses identity (chiplet
    // swizzle HARMFUL under uneven sparsity: 0.83x); dense uses head-first chiplet swizzle (co-locate on one XCD's L2,
    // ~+1.4%).
    int q_block, bh_index;
    if constexpr (Split) {
        // One CTA per part: its (q_block,bh) come from the uploaded per-part schedule entry.
        const int4 sched_entry = tail_sched[blockIdx.x];
        q_block = sched_entry.x;
        bh_index = sched_entry.y;
    } else if (grid_nqb) {
        // 1-D grid. lite's lpt_order runs heaviest q-blocks first (null=identity); pure grid-slot→WG reorder →
        // bit-identical. Radix is grid_nqb (TRUE #q-blocks), not gridDim.x (prefix/split's x-dim is #parts).
        const int wg = lpt_order ? lpt_order[wg_id_raw] : wg_id_raw;
        q_block = wg % grid_nqb;
        bh_index = wg / grid_nqb;
    } else {
        const int wg_id = isLite ? wg_id_raw : chiplet_swizzle(wg_id_raw, num_wgs, num_q_blocks);
        q_block = wg_id % num_q_blocks;
        bh_index = wg_id / num_q_blocks;
    }

    // Q-tile ownership: each wave owns QTilesPerWave(=3) 16-row q-tiles. q_tile_start_row = tile's origin seq-row, lane
    // `col` reads/writes row (start+col). Parallel arrays indexed by q_group; valid[] masks past-seq-end.
    const int num_q_tiles = (seq_len + QTileRows - 1) / QTileRows; // ceil
    bool valid[QTilesPerWave];
    int q_tile_start_row[QTilesPerWave];
    #pragma unroll
    for (int q_group = 0; q_group < QTilesPerWave; ++q_group) {
        const int q_tile = QTileGrid(q_block, wave_id, q_group);
        valid[q_group] = q_tile < num_q_tiles;
        q_tile_start_row[q_group] = valid[q_group] ? q_tile * QTileRows : 0;
    }

    // ceil block count; last_kv_rows ∈ [1,64] = valid rows in the last (partial) K/V block.
    const int num_kv_blocks = (seq_len_kv + KvBlockRows - 1) / KvBlockRows;
    const int last_kv_rows = seq_len_kv - (num_kv_blocks - 1) * KvBlockRows;
    // KV stride/offset over [0,num_kv_blocks). Dense Split part walks every split_ways-th block from its offset; everyone
    // else (incl all lite) walks every block (stride 1, off 0).
    const int kv_stride = Split ? split_ways : 1;
    const int kv_offset = Split ? (blockIdx.x % split_ways) : 0;

    // Cross-attention offsets. Q/K/Out keep TRUE seq lens, SRD bounded to valid byte extent: partial last q-tile/K-block
    // reads 0 / drops store IN HARDWARE (zero-VGPR OOB). V_t padded to num_kv_blocks·64 rows (tail zeroed). Only Q/K/Out
    // addressing is layout-dependent.
    const auto QOLayout = seq_layout<kBSHD, HeadDim>(heads);     // Q/Out (seq, head-dim elems)
    const auto KHbmDword = seq_layout<kBSHD, KRowDwords>(heads); // K     (seq, dwords)
    const int k_row_dwords = kBSHD ? heads * KRowDwords : KRowDwords;
    long long bh_offset_q, bh_offset_kv;
    if constexpr (kBSHD) {
        const long long b = bh_index / heads, h = bh_index % heads;
        bh_offset_q = b * (long long)seq_len * heads * HeadDim + h * HeadDim;
        bh_offset_kv = b * (long long)seq_len_kv * heads * HeadDim + h * HeadDim;
    } else {
        bh_offset_q = (long long)bh_index * seq_len * HeadDim;
        bh_offset_kv = (long long)bh_index * seq_len_kv * HeadDim;
    }
    const long long bh_offset_v = (long long)bh_index * num_kv_blocks * KvBlockRows * HeadDim;
    // BHSD: per-head contiguous (seq·HeadDim·2). BSHD: rows interleaved → ×heads span.
    const uint32_t q_bytes = (uint32_t)((long long)seq_len * (kBSHD ? heads : 1) * HeadDim * 2);
    const uint32_t kv_bytes = (uint32_t)((long long)seq_len_kv * (kBSHD ? heads : 1) * HeadDim * 2);
    // V_t per-head-contiguous (padded to num_kv_blocks·64 rows). Bound SRD to that span so a stray pv offset CLAMPS to 0,
    // not faults.
    const uint32_t v_bytes = (uint32_t)((long long)num_kv_blocks * KvBlockRows * HeadDim * 2);
    const auto q_buffer = make_buffer(Q + bh_offset_q, q_bytes);
    const auto k_buffer = make_buffer(K + bh_offset_kv, kv_bytes);
    const auto v_buffer = make_buffer(V + bh_offset_v, v_bytes);
    const auto out_buffer = make_buffer(Out + bh_offset_q, q_bytes);

    constexpr int KLdsBufDwords = KvBlockRows * KLdsStrideDwords; // 4096 dwords/buffer (64 rows × 64)
    __shared__ uint32_t lds_k[2][KLdsBufDwords];                  // 32 KB (2-buf, stride 64)
    // V_t consumed directly from L1 (no LDS staging). Q-park: qg0,1 in VGPR; qg2 (all 8 d-tiles) in one 32 KB LDS array
    // (Q2Lds, d-tile t @ slot t). 32 KB K + 32 KB qg2 = exactly 64 KB LDS; qg2 reads (lgkmcnt) overlap pv's V loads
    // (vmcnt).
    __shared__ uint32_t lds_q2[Q2LdsDwords];     // 32 KB (entire qg2, d-tiles 0..7; lite: 0..6)
    auto load_q2 = [&](int d_tile) -> bf16x4_t { // d-tile d_tile of qg2 for this lane (d_tile < last for lite)
        return *reinterpret_cast<const bf16x4_t *>(&lds_q2[Q2Lds(wave_id, d_tile, lane)]);
    };
    // LITE skip-vote ring, reusing the evicted last-d-tile lds_q2 slot. Per-wave region → no cross-wave contention; dword
    // (0/1) avoids sub-dword RMW.
    auto sverdict = [&](int w, int r) -> uint32_t & { return lds_q2[Q2Lds(w, HeadDimTiles - 1, 0) + r]; };

    // K[kv_block] HBM→LDS into double-buffer half `k_buf` (group B). Wave w owns rows [w·16,+16). XOR swizzle on the HBM
    // SOURCE: lane L fetches dword of swizzled 8B unit (L>>1)^c, in-unit dword L&1, c=kv_row&0xF. LDS contiguous → unit p
    // holds HBM (p^c), matches qk read.
    constexpr uint32_t KRowBytes = (uint32_t)(KLdsStrideDwords * 4);
    // K block stride (bytes) = KvBlockRows × seq-row stride. BHSD folds to 16384; BSHD is ×heads (rows interleaved).
    const uint32_t KBlockBytes = (uint32_t)(KvBlockRows * k_row_dwords * 4);
    const int wave_row0 = KLoadRow(wave_in_group, 0);          // first kv-row this load-wave owns
    const uint32_t k_row_bytes = (uint32_t)(k_row_dwords * 4); // per-seq-row HBM byte stride
    auto direct_lds_k = [&](int kv_block, int k_buf) {
        const uint32_t lds_row0 = to_scalar(lds_addr(&lds_k[k_buf][KLdsDword(wave_row0, 0)]));
        const uint32_t hbm_block = to_scalar((uint32_t)((uint64_t)kv_block * KBlockBytes));
        if constexpr (kBSHD) {
            // BSHD seq-row stride is a runtime heads·KRowDwords MULTIPLY → per-lane offen VGPR can't be remat → LLVM pins all
            // 16 per-c values and spills 4 EVERY iter. Fix: scalar soffset = hbm_block + (wave_row0+c)·stride·4
            // (wave-uniform) + cheap per-lane offen = swz_lane·4 = (lane·4) XOR (c·8) → 0 spill.
            const uint32_t lane_x4 = (uint32_t)(lane << 2); // per-lane base
            const uint32_t base_soff = to_scalar(hbm_block + (uint32_t)wave_row0 * k_row_bytes);
            #pragma unroll
            for (int c = 0; c < KRowsPerLoadWave; ++c) {
                const uint32_t soff = to_scalar(base_soff + (uint32_t)c * k_row_bytes);
                // == (lane^(c<<1))<<2: XOR distributes through <<2, and (c<<1)<<2 = c<<3, so this is swz_lane·4.
                const uint32_t off = lane_x4 ^ (uint32_t)(c << 3);
                buffer_load_b32_to_lds(k_buffer, lds_row0 + c * KRowBytes, off, soff);
            }
        } else {
            // BHSD: static stride → kv_row·stride·4 folds to a shift LLVM remats for free (0 spill). Keep single-VGPR-offset
            // form byte-identical.
            #pragma unroll
            for (int c = 0; c < KRowsPerLoadWave; ++c) {
                const int kv_row = KLoadRow(wave_in_group, c);
                // 8B-unit = (lane>>1)^c, in-unit dword = lane&1. e.g. lane=5,c=2: unit=(2^2)=0, dword=1.
                const uint32_t swz_lane = (uint32_t)((((lane >> 1) ^ c) << 1) | (lane & 1));
                const uint32_t hbm_byte = (uint32_t)((KHbmDword(kv_row, 0) + swz_lane) * 4);
                buffer_load_b32_to_lds(k_buffer, lds_row0 + c * KRowBytes, hbm_byte, hbm_block);
            }
        }
    };

    // ── LiteAttention skip machinery (lite only; compiled away for dense) ─────────
    // [&]-lambdas over locals: a read-list cursor (which K-blocks this step computes) + a write-list emitter (next-step
    // RLE + MustDoList force-compute). ── Read-list cursor: which K-blocks this step computes ──────────────────────
    // Skip-list row-major [bh,q_block,num_k_blocks]; lbase = CTA's row origin. read_list[lbase]=RLE length then ranges.
    // lstep = reverse-list dir (phase 1 asc, 0 desc). rl() reads via to_scalar → cursor stays in SGPRs, off the MFMA VGPR
    // budget.
    const int lstep = phase ? 1 : -1;
    long long lbase = 0;
    int list_len = 0;
    auto rl = [&](int idx) -> int { return to_scalar((int)read_list[lbase + idx]); };

    // Cursor state: range_block=current kept block; range_end=exclusive walk-end; rle_idx=next marker pair;
    // range_cont=last step stayed in range (drives record()'s range-end marker). Honors kv_stride/kv_offset, but lite is
    // never Split → stride 1.
    int rle_idx = 0, range_block = 0, range_end = 0;
    bool range_cont = false;
    auto open_range = [&]() { // load the next range at rle_idx (always non-empty)
        range_block = rl(rle_idx) + lstep;
        range_end = rl(rle_idx - 1) + lstep;
        rle_idx -= 2;
    };
    auto next_kept = [&](int cur) -> int { // next kept block after `cur` (one step), -1 if done
        const int next_blk = cur + lstep;
        if ((lstep > 0) ? (next_blk < range_end) : (next_blk > range_end)) {
            range_block = next_blk;
            range_cont = true;
            return next_blk;
        }
        range_cont = false;
        if (rle_idx >= 1) {
            open_range();
            return range_block;
        }
        return -1;
    };
    auto advance = [&](int cur, int n) -> int { // skip `n` kept blocks (stride), -1 if exhausted
        for (int s = 0; s < n && cur >= 0; ++s)
            cur = next_kept(cur);
        return cur;
    };

    // ── Write-list emitter: next-step skip list (tid 0 only) ─────────────────────

    int write_idx = 1;      // output cursor (slot 0 holds the final length)
    bool skip_state = true; // running skip/compute state, for transition detection
    bool prev_ended_range = false;

    // MustDoList: global force-compute RLE [len,s0,e0,…] of [start,end) K-blocks pinned "compute" (vote can't drop them,
    // e.g. sinks). Walked monotonically in lstep order; nullptr/len-0 ⇒ dormant (byte-identical to plain skip).
    const bool have_md = isLite && (must_do_list != nullptr) && ((int)must_do_list[0] > 0);
    const int md_nr = have_md ? ((int)must_do_list[0] >> 1) : 0;      // # of [s,e) ranges
    int md_ri = (lstep > 0) ? 0 : (md_nr - 1);                        // range cursor (walks in lstep dir)
    auto md_lo = [&](int r) { return (int)must_do_list[1 + 2 * r]; }; // range r start (inclusive)
    auto md_hi = [&](int r) { return (int)must_do_list[2 + 2 * r]; }; // range r end   (exclusive)
    auto md_in_cur = [&](int b) { return md_ri >= 0 && md_ri < md_nr && b >= md_lo(md_ri) && b < md_hi(md_ri); };
    // Is `block` forced-compute? Advance the monotonic cursor past ranges we've walked off, then test.
    auto md_force = [&](int block) -> bool {
        if (!have_md)
            return false;
        if (lstep > 0)
            while (md_ri < md_nr && block >= md_hi(md_ri))
                ++md_ri; // ascending: drop ranges behind us
        else
            while (md_ri >= 0 && block < md_lo(md_ri))
                --md_ri; // descending
        return md_in_cur(block);
    };
    // block's completed vote: AND the 8 wave votes (skip iff all voted skip); MustDoList pins → compute.
    auto vote = [&](int block, int verdict_step) -> bool {
        int sk = 1;
        #pragma unroll
        for (int w = 0; w < NumWaves; ++w)
            sk &= (int)sverdict(w, verdict_step & (VerdictRing - 1));
        bool skip = (sk != 0);
        if (md_force(block))
            skip = false; // MustDoList: pin as compute
        return skip;
    };
    // RLE-encode the vote inline: a marker on each skip<->compute flip, plus a range-end marker when a kept range closes.
    auto record = [&](int block, int verdict_step, bool is_range_end) {
        bool skip = vote(block, verdict_step);
        if (skip != skip_state) {
            write_list[lbase + write_idx++] = (int16_t)block;
            skip_state = skip;
        }
        if (is_range_end) {
            skip_state = true;
            if (!skip)
                write_list[lbase + write_idx++] = (int16_t)(block + lstep);
        }
    };

    // Prologue: first processed K-block → LDS (group B only; A idle until iter 1). Dense starts at kv_offset; lite
    // advances the cursor kv_offset KEPT blocks in.
    int first_blk = kv_offset;
    bool has_any_block = (kv_offset < num_kv_blocks);
    if constexpr (isLite) {
        const int num_k_blocks = num_kv_blocks + 2;              // RLE row width: [len, markers…]
        const int nqb_list = grid_nqb ? grid_nqb : num_q_blocks; // 1-D grids: gridDim.x ≠ #q-blocks
        lbase = ((long long)bh_index * nqb_list + q_block) * num_k_blocks;
        list_len = to_scalar((int)read_list[lbase]);
        rle_idx = list_len;
        if (list_len > 0) {
            open_range();
            first_blk = advance(range_block, kv_offset);
        }
        has_any_block = (first_blk >= 0);
    }
    if (!group_a && has_any_block)
        direct_lds_k(first_blk, /*k_buf=*/0); // step starts at 0 → slot 0

    // Persistent VGPR Q (no reload across K-blocks). Lane (q_tile_start_row+col) owns one q-row; each load_bf16x4 grabs
    // 4 head-dims of d-tile d (quad picks which 4 of 16). qg0,1 in VGPR; qg2 in lds_q2 (d-tile t @ slot t).
    bf16x4_t q_frag[2][HeadDimTiles];
    auto load_q = [&](int q_group, int d) -> bf16x4_t {
        return valid[q_group] ? load_bf16x4(q_buffer, QOLayout(q_tile_start_row[q_group] + col, HdFrag(d, quad)) * 2) : bf16x4_t{0, 0, 0, 0};
    };
    // NOTE: keep qg0/qg1 as SEPARATE prologue load loops — collapsing into one `for (g)` loop re-introduces a VGPR spill
    // (allocator coloring is load-structure-sensitive). The qk-phase qg0/qg1 collapse is safe; this prologue one is not.
    #pragma unroll
    for (int d = 0; d < HeadDimTiles; ++d)
        q_frag[0][d] = load_q(0, d);
    #pragma unroll
    for (int d = 0; d < HeadDimTiles; ++d)
        q_frag[1][d] = load_q(1, d);
    // qg2: all 8 d-tiles → lds_q2 (d-tile d @ slot d), HBM→VGPR→LDS. LITE: LAST d-tile stays in VGPR (q2_last), freeing
    // its 4 KB slot for the skip-vote ring; dense parks all 8.
    bf16x4_t q2_last{0, 0, 0, 0};
    #pragma unroll
    for (int d = 0; d < HeadDimTiles; ++d) {
        bf16x4_t v = load_q(2, d);
        if constexpr (isLite) {
            if (d == HeadDimTiles - 1) {
                q2_last = v;
                continue;
            }
        }
        u32x2_t raw;
        __builtin_memcpy(&raw, &v, 8);
        const uint32_t off = Q2Lds(wave_id, d, lane);
        lds_q2[off + 0] = raw[0];
        lds_q2[off + 1] = raw[1];
    }
    __syncthreads(); // lds_q2 visible to all reads below

    fp32x4_t o_acc[QTilesPerWave][HeadDimTiles];
    #pragma unroll
    for (int q_group = 0; q_group < QTilesPerWave; ++q_group)
        #pragma unroll
    for (int d = 0; d < HeadDimTiles; ++d)
        o_acc[q_group][d] = zero_fp32x4();
    float row_max[QTilesPerWave], row_sum[QTilesPerWave];
    #pragma unroll
    for (int q_group = 0; q_group < QTilesPerWave; ++q_group) {
        row_max[q_group] = -FLT_MAX;
        row_sum[q_group] = 0.f;
    }

    // Loop-carried per-wave state: scores written by qk_phase before softmax reads, probs by softmax_update before pv.
    // Zero-init is logically dead but LOAD-BEARING: one clean def point for the allocator; dropping it respills 64 VGPR.
    fp32x4_t scores[QTilesPerWave][KvSubtiles];
    bf16x4_t probs[QTilesPerWave][KvSubtiles];
    #pragma unroll
    for (int q_group = 0; q_group < QTilesPerWave; ++q_group)
        #pragma unroll
    for (int kv_sub = 0; kv_sub < KvSubtiles; ++kv_sub) {
        scores[q_group][kv_sub] = zero_fp32x4();
        probs[q_group][kv_sub] = bf16x4_t{0, 0, 0, 0};
    }

    auto qk_phase = [&](int k_buf, fp32x4_t scores[QTilesPerWave][KvSubtiles]) {
        // Swizzled LDS read addr = OR of disjoint bit-fields (OR≡add): qk_swz_base[11:3] | (d_pair^col_hi)<<5 (bank
        // swizzle) | kv_sub<<12 | k_buf<<14 (which of 2 K LDS halves). d_pair^col_hi is the only per-read compute.
        const uint32_t lane_off = (uint32_t)k_buf << 14;
        auto k_addr = [&](int kv_sub, int d_pair) -> uint32_t { return qk_swz_base | (((uint32_t)d_pair ^ col_hi) << 5) | ((uint32_t)kv_sub << 12) | lane_off; };

        // Loop i in [0,TotalReads) splits 3 ways: kv_sub=i%KvSubtiles (scores tile), d_pair=i/KvSubtiles (head-dim
        // col-pair, K+Q frag), slot=i%Depth (ring reg s0/s1/s2). e.g. i=5 → kv_sub=1,d_pair=1,slot=2. Depth(3) coprime with
        // KvSubtiles(4) so the ring rotates independently.
        constexpr int TotalReads = KvSubtiles * 4; // 16 ds_read2 = 4 kv-subtiles × 4 d-pairs
        constexpr int Depth = 3;                   // K-tile prefetch ring depth
        constexpr int Q2Depth = 2;                 // qg2-from-LDS ping-pong depth
        mfma_prio_hi();
        sched_fence();

        // K pre-issues (depth-3 ring): each ds_read2 fetches a d-tile PAIR (lo @offset0, hi @offset1).
        u32x4_t s0 = ds_read2_b64_imm<0, 16>(k_addr(0, 0));
        u32x4_t s1 = ds_read2_b64_imm<0, 16>(k_addr(1, 0));
        u32x4_t s2 = ds_read2_b64_imm<0, 16>(k_addr(2, 0));

        // qg2 (only LDS-resident q-group): depth-2 ping-pong per head-dim half h (h=0→d-tiles 0..3, h=1→4..7). q2[h][slot]
        // = d-tile slot+h·Q2HalfTiles.
        bf16x4_t q2[2][Q2Depth];
        #pragma unroll
        for (int p = 0; p < Q2Depth; ++p)
            #pragma unroll
        for (int h = 0; h < 2; ++h)
            q2[h][p] = load_q2(p + h * Q2HalfTiles);

        // Q fragment for one qk MFMA (g, half h, col-pair d_pair): qg0/qg1 in VGPR (q_frag); qg2 streams LDS ping-pong
        // (q2[h][slot]) — except lite-only qg2 last d-tile, evicted to VGPR (q2_last) so its LDS slot holds the skip-vote
        // ring.
        auto q_fragment = [&](int g, int h, int d_pair, int q2_slot) -> bf16x4_t {
            if (g < 2)
                return q_frag[g][d_pair + h * Q2HalfTiles];
            if constexpr (isLite)
                if (h == 1 && d_pair == Q2HalfTiles - 1)
                    return q2_last;
            return q2[h][q2_slot];
        };

        #pragma unroll
        for (int i = 0; i < TotalReads; ++i) {
            const int kv_sub = i % KvSubtiles;
            const int d_pair = i / KvSubtiles;
            const int slot = i % Depth;
            u32x4_t &cur = (slot == 0) ? s0 : (slot == 1) ? s1 : s2;
            if (i == TotalReads - 1)
                wait_lgkm<0>(cur);
            else
                wait_lgkm<Depth - 1>(cur);
            // Contract d-tile pair {lo=d_pair, hi=d_pair+Q2HalfTiles} into every q-group's scores: K from this ring slot, Q
            // from q_frag (qg0,1) or q2 (qg2).
            const bf16x4_t k_pair[2] = {lo2(cur), hi2(cur)};
            const int q2_slot = d_pair % Q2Depth; // qg2 ping-pong slot keyed on d_pair (all 4 kv_sub of one d_pair reuse the same qg2 read)
            #pragma unroll
            for (int g = 0; g < QTilesPerWave; ++g)
                #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const bf16x4_t qf = q_fragment(g, h, d_pair, q2_slot);
                // First MFMA into this scores tile inits from zero (no separate memset); later d_pair/h accumulate in place.
                if (d_pair == 0 && h == 0)
                    scores[g][kv_sub] = asm_mfma_zero(k_pair[0], qf);
                else
                    asm_mfma(k_pair[h], qf, scores[g][kv_sub]);
            }
            // Prefetch the next K pair into this ring slot (depth-3 ahead).
            if (i + Depth < TotalReads) {
                const int next_i = i + Depth;
                cur = ds_read2_b64_imm<0, 16>(k_addr(next_i % KvSubtiles, next_i / KvSubtiles));
            }
            // Prefetch the next qg2 d-tile pair (d_pair+Q2Depth) into its ping-pong slot.
            if (kv_sub == KvSubtiles - 1 && d_pair + Q2Depth < Q2HalfTiles) {
                const int pf_d_pair = d_pair + Q2Depth; // d_pair being prefetched
                #pragma unroll
                for (int h = 0; h < 2; ++h) {
                    // LITE: the last d-tile (h=1, top d_pair) lives in q2_last (VGPR), not LDS.
                    if constexpr (isLite) {
                        if (h == 1 && pf_d_pair == Q2HalfTiles - 1)
                            continue;
                    }
                    q2[h][q2_slot] = load_q2(pf_d_pair + h * Q2HalfTiles);
                }
            }
        }
        mfma_prio_lo();
    };

    // ─── PV: O += V·Pᵀ via mfma(A=V_t, B=P). 16 super-chunks/iter, each lane load_u32x4s its
    // pre-packed bf16x8 from V_t (VtLayout). Depth-4 prefetch ring. chunk → hd-tile chunk/2 +
    // kv-subtile pair; all q-groups share the V operand. ────────────────────────
    auto chunk_byte_off = [&](int kv_iter, int chunk) -> uint32_t {
        return (uint32_t)(VtLayout(kv_iter, chunk, lane) * 2); // bf16 elems → bytes
    };

    constexpr int PvDepth = 4; // V_t prefetch ring depth (VtChunks=16 super-chunks / kv-iter)
    // verdict_step = processing step of the block being pv'd (verdict-ring slot; ignored !isLite). Part B soft
    // block-skip: a "skip" wave does NO pv (its probs were zeroed in softmax → result unchanged), dropping 0-adding MFMAs
    // + V_t loads. Vote is in LDS from this wave's softmax (A: prev iter P2; B: this iter P1), visible across record()'s
    // barriers.
    auto pv_phase_preload = [&](int kv_iter, const bf16x4_t probs[QTilesPerWave][KvSubtiles], int verdict_step, u32x4_t v_ring[PvDepth]) {
        if constexpr (isLite) {
            if (sverdict(wave_id, verdict_step & (VerdictRing - 1)))
                return;
        }
        #pragma unroll
        for (int p = 0; p < PvDepth; ++p)
            v_ring[p] = load_u32x4(v_buffer, (int)chunk_byte_off(kv_iter, p));
    };
    auto pv_phase_compute = [&](int kv_iter, const bf16x4_t probs[QTilesPerWave][KvSubtiles], int verdict_step, u32x4_t v_ring[PvDepth]) {
        if constexpr (isLite) {
            if (sverdict(wave_id, verdict_step & (VerdictRing - 1)))
                return;
        }
        mfma_prio_hi();
        sched_fence();
        #pragma unroll
        for (int chunk = 0; chunk < VtChunks; ++chunk) {
            // chunk in [0,16): d_tile=chunk/2 (hd-tile 0..7); kv_sub0=(chunk&1)*2 (kv-subtile pair base, even→{0,1}
            // odd→{2,3}). e.g. chunk=5 → d_tile=2, kv_sub0=2.
            const int d_tile = chunk >> 1;
            const int kv_sub0 = (chunk & 1) * 2;
            u32x4_t &cur = v_ring[chunk % PvDepth];
            // one b128 V load = a kv-subtile pair; v_pair[h] is the operand for kv-subtile kv_sub0+h.
            const bf16x4_t v_pair[2] = {lo2(cur), hi2(cur)};
            // Issue the next ring V-load BEFORE this chunk's MFMAs (latency lead).
            if (chunk + PvDepth < VtChunks)
                cur = load_u32x4(v_buffer, (int)chunk_byte_off(kv_iter, chunk + PvDepth));
            // v_pair[h] feeds all q-groups back-to-back before the next half — shorter V-reg lifetime than q-group-outer
            // ordering.
            #pragma unroll
            for (int h = 0; h < 2; ++h)
                #pragma unroll
            for (int q_group = 0; q_group < QTilesPerWave; ++q_group)
                asm_mfma(v_pair[h], probs[q_group][kv_sub0 + h], o_acc[q_group][d_tile]);
        }
        sched_fence();
        mfma_prio_lo();
    };

    // L1 prefetcher for V_t[kv_iter_next], fired from Group A's softmax-only P2 (vmcnt per-wave → independent of
    // consumers' load_u32x4). Wave-split 4×4 = 16 unique cache-line requests hit all L1 sets. Dest v[140:143] = discard
    // sink, LAST op before end-of-iter wait_vmcnt0.
    auto prefetch_v_t_l1 = [&](int kv_iter_next) {
        if (kv_iter_next < 0)
            return;
        #pragma unroll
        for (int i = 0; i < VtChunks / LoadWaves; ++i) {
            const int chunk = VtPrefetch(wave_in_group, i);
            const uint32_t chunk_off = (uint32_t)(VtLayout(kv_iter_next, chunk, lane) * 2);
            // v[140:143] = discard sink (L1-fill side effect only; data never read). A fixed GPR keeps it out of the
            // allocator's live set.
            asm volatile("buffer_load_dwordx4 v[140:143], %0, %1, 0 offen" : : "v"(chunk_off), "s"(v_buffer) : "v140", "v141", "v142", "v143", "memory");
        }
    };

    auto softmax = [&](fp32x4_t scores[QTilesPerWave][KvSubtiles], bf16x4_t probs[QTilesPerWave][KvSubtiles], int step, int kv_block) {
        // Per-q-group block max — local, not loop-carried.
        float bmax_local[QTilesPerWave];
        // Partial last K/V block (seq_len_kv%64≠0): K-OOB rows read 0 → score 0, polluting denom. Force kv-rows ≥
        // last_kv_rows to -inf (exp2→0). No-op if aligned.
        if (last_kv_rows < KvBlockRows && kv_block == num_kv_blocks - 1) {
            const int row0 = quad * 4;
            #pragma unroll
            for (int kv_sub = 0; kv_sub < KvSubtiles; ++kv_sub)
                #pragma unroll
            for (int i = 0; i < 4; ++i)
                if (kv_sub * MfmaTile + row0 + i >= last_kv_rows) {
                    scores[0][kv_sub][i] = -FLT_MAX;
                    scores[1][kv_sub][i] = -FLT_MAX;
                    scores[2][kv_sub][i] = -FLT_MAX;
                }
        }
        // Per-lane bmax fold runs HERE (not qk_phase). Softmax runs in P2 while the other group is in pv+qk → the v_max3
        // VALU chain hides under their MFMA writeback.
        #pragma unroll
        for (int q_group = 0; q_group < QTilesPerWave; ++q_group) {
            float bm = -FLT_MAX;
            #pragma unroll
            for (int kv_sub = 0; kv_sub < KvSubtiles; ++kv_sub) {
                bm = fmaxf(bm, fmaxf(fmaxf(scores[q_group][kv_sub][0], scores[q_group][kv_sub][1]), fmaxf(scores[q_group][kv_sub][2], scores[q_group][kv_sub][3])));
            }
            bmax_local[q_group] = bm;
        }
        // LITE skip vote: keep iff some valid q-row's block-max is within thr (log2) of its running max. A skipping wave
        // zeros its probs; record() ANDs the 8 votes.
        if constexpr (isLite) {
            bool any_keep = false;
            #pragma unroll
            for (int q_group = 0; q_group < QTilesPerWave; ++q_group) {
                const float block_max_log2 = bmax_local[q_group] * scale_log2e;    // raw block-max → log2 units
                const bool within_thr = (block_max_log2 - row_max[q_group]) > thr; // thr<0: gap above |thr| ⇒ keep
                any_keep = any_keep || (valid[q_group] && within_thr);
            }
            bool skip_wave = !__any(any_keep); // skip only if NO lane in the wave voted keep
            if (step == 0)
                skip_wave = false; // never skip the first processed block
            if (lane == 0)
                sverdict(wave_id, step & (VerdictRing - 1)) = skip_wave ? 1u : 0u;
            if (skip_wave) {
                // Zero probs so the skipped pv adds nothing; the keep predicate guarantees this block never raises the running
                // max → state stays correct.
                #pragma unroll
                for (int q_group = 0; q_group < QTilesPerWave; ++q_group)
                    #pragma unroll
                for (int kv_sub = 0; kv_sub < KvSubtiles; ++kv_sub)
                    probs[q_group][kv_sub] = bf16x4_t{0, 0, 0, 0};
                return;
            }
        }
        // ── Fold + online update, per q-group (free function → scoped temporaries). ──
        #pragma unroll
        for (int q_group = 0; q_group < QTilesPerWave; ++q_group)
            softmax_update(scores[q_group], o_acc[q_group], probs[q_group], row_max[q_group], row_sum[q_group], scale_log2e, bmax_local[q_group]);
    };

    // ─── Main loop ─────────────────────────────────────────────────────────
    //   Group A  P1: pv(V_t[N-1]) + qk(K[N])      P2: softmax + V_t prefetch
    //   Group B  P1: K[N+1] HBM→LDS + softmax     P2: pv(V_t[N-1]) + qk(K[N])
    wait_vmcnt0();
    lds_barrier();
    // ─── Pipeline over K-blocks (forward 0..num_kv_blocks-1) ────────────────────
    // `step` = step counter (K double-buffer parity); kv_block/prev_block qk'd/pv'd; next_block streamed by B. Dense+lite
    // share ONE body: dense kv_block follows the arithmetic stride; lite's cursor walks kept blocks (stride 1) + tid 0
    // records the write list (`if constexpr`'d out for dense).
    int prev_block = -1, step = 0; // step counts PROCESSED blocks (drives K double-buffer parity)
    int kv_block = first_blk;      // current processed block
    bool has_work = has_any_block;
    u32x4_t a_v_ring[PvDepth];
    if (group_a) {
        if (!has_work) return;
        {
            // Next processed block: lite cursor (every kv_stride-th KEPT block) or dense stride. block_closed_range = "kv_block
            // ended its RLE range" → record's flush.
            int next_block;
            bool has_next;
            if constexpr (isLite) {
                next_block = advance(kv_block, kv_stride);
                has_next = (next_block >= 0);
            } else {
                next_block = kv_block + kv_stride;
                has_next = (next_block < num_kv_blocks);
            }
            const bool block_closed_range = isLite ? !range_cont : false;
            const int k_buf_cur = step & 1, k_buf_next = (step + 1) & 1;
            qk_phase(k_buf_cur, scores);

            __builtin_amdgcn_s_barrier();

            softmax(scores, probs, step, kv_block);
            // tid 0 folds prev_block's skip verdict into the write-list RLE. Safe: vote stored to sverdict last iter
            // (VerdictRing depth 2 keeps it live).
            if (has_next) {
                prefetch_v_t_l1(kv_block);
                pv_phase_preload(kv_block, probs, step - 1 + 1, a_v_ring);
            }
        
            // End-of-iter barrier: vmcnt(0) drains direct_lds_k writes (buffer_load…lds is VMCNT-tracked, not LGKMCNT) so next
            // qk can't read stale K; lds_barrier covers LGKMCNT.
            wait_vmcnt0();
            lds_barrier();
            prev_block = kv_block;
            ++step;
            prev_ended_range = block_closed_range;
            kv_block = next_block;
            has_work = has_next;
        }
        while (has_work) {
            // Next processed block: lite cursor (every kv_stride-th KEPT block) or dense stride. block_closed_range = "kv_block
            // ended its RLE range" → record's flush.
            int next_block;
            bool has_next;
            if constexpr (isLite) {
                next_block = advance(kv_block, kv_stride);
                has_next = (next_block >= 0);
            } else {
                next_block = kv_block + kv_stride;
                has_next = (next_block < num_kv_blocks);
            }
            const bool block_closed_range = isLite ? !range_cont : false;
            const int k_buf_cur = step & 1, k_buf_next = (step + 1) & 1;
            pv_phase_compute(prev_block, probs, step - 1, a_v_ring);
            qk_phase(k_buf_cur, scores);

            __builtin_amdgcn_s_barrier();

            softmax(scores, probs, step, kv_block);
            // tid 0 folds prev_block's skip verdict into the write-list RLE. Safe: vote stored to sverdict last iter
            // (VerdictRing depth 2 keeps it live).
            if constexpr (isLite) {
                if (tid == 0)
                    record(prev_block, step - 1, prev_ended_range);
            }
            if (has_next) {
                prefetch_v_t_l1(kv_block);
                pv_phase_preload(kv_block, probs, step - 1 + 1, a_v_ring);
            }
            
            // End-of-iter barrier: vmcnt(0) drains direct_lds_k writes (buffer_load…lds is VMCNT-tracked, not LGKMCNT) so next
            // qk can't read stale K; lds_barrier covers LGKMCNT.
            wait_vmcnt0();
            lds_barrier();
            prev_block = kv_block;
            ++step;
            prev_ended_range = block_closed_range;
            kv_block = next_block;
            has_work = has_next;
        }
        // Epilogue. The loop defers each block's pv (and B's softmax) by one iter, so prev_block is still pending: B owes a
        if (step > 0) {
            u32x4_t v_ring[PvDepth];
            pv_phase_preload(prev_block, probs, step - 1, v_ring);
            pv_phase_compute(prev_block, probs, step - 1, v_ring);
        }
        if constexpr (isLite) {
            lds_barrier(); // group B's last-block vote → visible to tid 0
            if (tid == 0 && step > 0) {
                record(prev_block, step - 1, /*is_range_end=*/true); // finalize inline RLE
                write_list[lbase] = (int16_t)(write_idx - 1);        // RLE length
            }
        }

        // ─── Output ────────────────────────────────────────────────────────────
        // row_sum summed across the 4 quad-lanes once here (math-equiv, rescale is row-global). inv_sum=0 when row_sum==0
        // prevents 0×inf=NaN.
        #pragma unroll
        for (int q_group = 0; q_group < QTilesPerWave; ++q_group)
            row_sum[q_group] = cross_lane_sum(row_sum[q_group]);
        if constexpr (Split) {
            // A split part exports raw fp32 partials (o, plus row max `m` and sum `l`), UNSCALED — o/l division + bf16 pack are
            // merge_parts' job. One CTA == one part.
            TailScratch parts(part_buf);
            const int part = blockIdx.x;
            #pragma unroll
            for (int q_group = 0; q_group < QTilesPerWave; ++q_group) {
                // Absolute row of this (wave, q_group, lane) within the part's 384 rows.
                const int row = (wave_id * QTilesPerWave + q_group) * QTileRows + col;
                #pragma unroll
                for (int t = 0; t < HeadDimTiles; ++t) // store each head-dim fragment (4 fp32) of the o accumulator
                    *reinterpret_cast<fp32x4_t *>(&parts.o[PartO(part, row, HdFrag(t, quad))]) = o_acc[q_group][t];
                if (quad == 0) {                                        // m and l are per-row scalars — only the first quad-lane of the row writes them
                    parts.ml[PartMl(part, row)] = row_max[q_group];     // m = running softmax max
                    parts.ml[PartMl(part, row) + 1] = row_sum[q_group]; // l = running softmax denominator
                }
            }
            return;
        }
        #pragma unroll
        for (int q_group = 0; q_group < QTilesPerWave; ++q_group) {
            if (!valid[q_group])
                continue;
            const float inv_sum = (row_sum[q_group] == 0.0f) ? 0.0f : fast_rcp(row_sum[q_group]);
            // No PV mfma follows → packed v_pk_mul_f32 is safe. MaybeNaN pack4 scrubs residual NaN/Inf in o_acc from
            // pathological inputs.
            #pragma unroll
            for (int t = 0; t < HeadDimTiles; ++t) {
                o_acc[q_group][t] = o_acc[q_group][t] * inv_sum;
                const bf16x4_t out = pack4_bf16</*MaybeNaN=*/true>(o_acc[q_group][t][0], o_acc[q_group][t][1], o_acc[q_group][t][2], o_acc[q_group][t][3]);
                store_bf16x4(out_buffer, QOLayout(q_tile_start_row[q_group] + col, HdFrag(t, quad)) * 2, out);
            }
        }
    }
    else{
        if (!has_work) return;
            {
                // Next processed block: lite cursor (every kv_stride-th KEPT block) or dense stride. block_closed_range = "kv_block
                // ended its RLE range" → record's flush.
                int next_block;
                bool has_next;
                if constexpr (isLite) {
                    next_block = advance(kv_block, kv_stride);
                    has_next = (next_block >= 0);
                } else {
                    next_block = kv_block + kv_stride;
                    has_next = (next_block < num_kv_blocks);
                }
                const bool block_closed_range = isLite ? !range_cont : false;
                const int k_buf_cur = step & 1, k_buf_next = (step + 1) & 1;
                if (has_next)
                    direct_lds_k(next_block, k_buf_next);

                __builtin_amdgcn_s_barrier();

                qk_phase(k_buf_cur, scores);
                
                // End-of-iter barrier: vmcnt(0) drains direct_lds_k writes (buffer_load…lds is VMCNT-tracked, not LGKMCNT) so next
                // qk can't read stale K; lds_barrier covers LGKMCNT.
                wait_vmcnt0();
                lds_barrier();
                prev_block = kv_block;
                ++step;
                prev_ended_range = block_closed_range;
                kv_block = next_block;
                has_work = has_next;
            }
            while (has_work) {
                // Next processed block: lite cursor (every kv_stride-th KEPT block) or dense stride. block_closed_range = "kv_block
                // ended its RLE range" → record's flush.
                int next_block;
                bool has_next;
                if constexpr (isLite) {
                    next_block = advance(kv_block, kv_stride);
                    has_next = (next_block >= 0);
                } else {
                    next_block = kv_block + kv_stride;
                    has_next = (next_block < num_kv_blocks);
                }
                const bool block_closed_range = isLite ? !range_cont : false;
                const int k_buf_cur = step & 1, k_buf_next = (step + 1) & 1;
                softmax(scores, probs, step - 1, prev_block);
                if (has_next)
                    direct_lds_k(next_block, k_buf_next);
                
                u32x4_t v_ring[PvDepth];
                pv_phase_preload(prev_block, probs, step - 1, v_ring);

                __builtin_amdgcn_s_barrier();

                pv_phase_compute(prev_block, probs, step - 1, v_ring);
                qk_phase(k_buf_cur, scores);
            
            // End-of-iter barrier: vmcnt(0) drains direct_lds_k writes (buffer_load…lds is VMCNT-tracked, not LGKMCNT) so next
                // qk can't read stale K; lds_barrier covers LGKMCNT.
                wait_vmcnt0();
                lds_barrier();
                prev_block = kv_block;
                ++step;
                prev_ended_range = block_closed_range;
                kv_block = next_block;
                has_work = has_next;
            }
            // Epilogue. The loop defers each block's pv (and B's softmax) by one iter, so prev_block is still pending: B owes a
            // softmax (A did its at P2), both owe the final pv. Lite then flushes the last block's range-end vote + RLE length.
            if ( step > 0)
                softmax(scores, probs, step - 1, prev_block);
            if (step > 0) {
                u32x4_t v_ring[PvDepth];
                pv_phase_preload(prev_block, probs, step - 1, v_ring);
                pv_phase_compute(prev_block, probs, step - 1, v_ring);
            }
            if constexpr (isLite) {
                lds_barrier(); // group B's last-block vote → visible to tid 0
                if (tid == 0 && step > 0) {
                    record(prev_block, step - 1, /*is_range_end=*/true); // finalize inline RLE
                    write_list[lbase] = (int16_t)(write_idx - 1);        // RLE length
                }
            }

            // ─── Output ────────────────────────────────────────────────────────────
            // row_sum summed across the 4 quad-lanes once here (math-equiv, rescale is row-global). inv_sum=0 when row_sum==0
            // prevents 0×inf=NaN.
            #pragma unroll
            for (int q_group = 0; q_group < QTilesPerWave; ++q_group)
                row_sum[q_group] = cross_lane_sum(row_sum[q_group]);
            if constexpr (Split) {
                // A split part exports raw fp32 partials (o, plus row max `m` and sum `l`), UNSCALED — o/l division + bf16 pack are
                // merge_parts' job. One CTA == one part.
                TailScratch parts(part_buf);
                const int part = blockIdx.x;
                #pragma unroll
                for (int q_group = 0; q_group < QTilesPerWave; ++q_group) {
                    // Absolute row of this (wave, q_group, lane) within the part's 384 rows.
                    const int row = (wave_id * QTilesPerWave + q_group) * QTileRows + col;
                    #pragma unroll
                    for (int t = 0; t < HeadDimTiles; ++t) // store each head-dim fragment (4 fp32) of the o accumulator
                        *reinterpret_cast<fp32x4_t *>(&parts.o[PartO(part, row, HdFrag(t, quad))]) = o_acc[q_group][t];
                    if (quad == 0) {                                        // m and l are per-row scalars — only the first quad-lane of the row writes them
                        parts.ml[PartMl(part, row)] = row_max[q_group];     // m = running softmax max
                        parts.ml[PartMl(part, row) + 1] = row_sum[q_group]; // l = running softmax denominator
                    }
                }
                return;
            }
            #pragma unroll
            for (int q_group = 0; q_group < QTilesPerWave; ++q_group) {
                if (!valid[q_group])
                    continue;
                const float inv_sum = (row_sum[q_group] == 0.0f) ? 0.0f : fast_rcp(row_sum[q_group]);
                // No PV mfma follows → packed v_pk_mul_f32 is safe. MaybeNaN pack4 scrubs residual NaN/Inf in o_acc from
                // pathological inputs.
                #pragma unroll
                for (int t = 0; t < HeadDimTiles; ++t) {
                    o_acc[q_group][t] = o_acc[q_group][t] * inv_sum;
                    const bf16x4_t out = pack4_bf16</*MaybeNaN=*/true>(o_acc[q_group][t][0], o_acc[q_group][t][1], o_acc[q_group][t][2], o_acc[q_group][t][3]);
                    store_bf16x4(out_buffer, QOLayout(q_tile_start_row[q_group] + col, HdFrag(t, quad)) * 2, out);
                }
            }
    }
}

} // namespace mmai_attention

extern "C" {
using namespace mmai_attention;
// IMPORTANT: V here must point to pre-transposed V_t (call launch_v_transpose first; V_t and V have identical byte
// size). Tail KV-split workspace (dense only): fp32 o + (m,l) for MaxParts parts. workspace==nullptr ⇒ no split.
size_t tail_workspace_bytes(void) { return (size_t)MaxParts * (PartOFloats + PartMlFloats) * sizeof(float); }

// Lite LPT order buffer: one int per work-group (q-block × batch·head). workspace==nullptr ⇒ plain full-grid lite.
size_t lite_workspace_bytes(int batch, int heads, int seq_len) {
    const int qt = (seq_len + QTileRows - 1) / QTileRows;
    const int nqb = (qt + QTilesPerWG - 1) / QTilesPerWG;
    return (size_t)batch * heads * nqb * sizeof(int);
}

int launch_attention_forward(const void *q, const void *k, const void *v, void *out, int batch, int heads, int seq_len, int seq_len_kv, int head_dim, int layout, void *workspace,
                             void *stream) {
    if (batch <= 0 || heads <= 0)
        return 1;
    if (seq_len <= 0)
        return 2; // any Q len (partial last q-tile SRD-padded)
    if (seq_len_kv <= 0)
        return 5; // any K/V len (partial last block masked)
    if (head_dim != HeadDim)
        return 3;

    const int bh = batch * heads;
    const int qt = (seq_len + QTileRows - 1) / QTileRows;
    const int nqb = (qt + QTilesPerWG - 1) / QTilesPerWG, nkb = (seq_len_kv + 63) / 64;
    const dim3 grid(nqb, bh);
    const float scale_log2e = Log2e / sqrtf((float)HeadDim);
    // layout: 0 = BHSD ([B,H,S,D]); 1 = BSHD ([B,S,H,D], read natively, no transpose).

    // Tail KV-split: recover the stranded fractional round (when a workspace is given). Prefix = full waves as one
    // ordinary launch; split parts + merge run the stranded round.
    const TailPlan tp = plan_tail_split(nqb, bh, nkb, workspace != nullptr, (hipStream_t)stream);
    if (tp.parts < 0)
        return 6;
    if (tp.tail) {
        launch_tail_split(layout, q, k, v, out, seq_len, seq_len_kv, scale_log2e, heads, nqb, workspace, tp, (hipStream_t)stream);
        return (int)hipGetLastError();
    }
    fwd_launch</*isLite=*/false, /*Split=*/false>(layout, grid, (hipStream_t)stream, q, k, v, out, seq_len, seq_len_kv, scale_log2e, heads);
    return (int)hipGetLastError();
}

// LiteAttention forward (Part A cross-timestep block-skip + Part B per-wave soft-skip). read/write_list = int16 RLE
// keep-lists [B,H,nqb,nkb+2] (read=skip_list[phase], write=[1-phase]); must_do_list = optional force-compute RLE
// (nullable); threshold log2 (~-10), phase = reverse dir, V pre-transposed. Verdict ring reuses evicted qg2 LDS slot →
// total LDS == dense's. Full-grid launch, each CTA writes its RLE inline; workspace (LPT buffer) runs q-blocks
// heaviest-first (bit-identical), else plain identity-order.
int launch_attention_forward_lite(const void *q, const void *k, const void *v, void *out, int batch, int heads, int seq_len, int seq_len_kv, int head_dim, const void *read_list,
                                  void *write_list, const void *must_do_list, float threshold, int phase, int layout, void *workspace, void *stream) {
    if (batch <= 0 || heads <= 0)
        return 1;
    if (seq_len <= 0)
        return 2; // any Q len (partial last q-tile masked by valid[])
    if (seq_len_kv <= 0)
        return 5; // any K/V len (partial last block masked in softmax)
    if (head_dim != HeadDim)
        return 3;
    if (!read_list || !write_list)
        return 4; // must_do_list is optional

    const int bh = batch * heads;
    const int qt = (seq_len + QTileRows - 1) / QTileRows;
    const int nqb = (qt + QTilesPerWG - 1) / QTilesPerWG, nkb = (seq_len_kv + 63) / 64;
    const dim3 grid(nqb, bh);
    const float scale_log2e = Log2e / sqrtf((float)HeadDim);

    // LPT work-balance reorder (default when a workspace is given and kv is long enough to amortize the ~few-µs sort):
    // run heaviest q-blocks first so HW dynamic dispatch finishes near the work-bound floor. Pure reorder →
    // bit-identical.
    if (workspace && nkb >= MinKvBlocksToSplit) {
        launch_lite_ordered(layout, q, k, v, out, seq_len, seq_len_kv, scale_log2e, heads, nqb, bh, nkb, read_list, write_list, must_do_list, threshold, phase, workspace,
                            (hipStream_t)stream);
        return (int)hipGetLastError();
    }

    // Lite runs the full grid as one launch (no tail KV-split — stranded round is sparse); each CTA walks its kept blocks
    // and RLE-encodes the write list inline.
    fwd_launch</*isLite=*/true, /*Split=*/false>(layout, grid, (hipStream_t)stream, q, k, v, out, seq_len, seq_len_kv, scale_log2e, heads, read_list, write_list, must_do_list,
                                                 threshold, phase);
    return (int)hipGetLastError();
}

// V transpose: V[seq_len,128] (bf16) → V_t in pv_phase's layout. Caller allocates V_t same byte size as V; n_rows =
// batch*heads*seq_len.
int launch_v_transpose(const void *v, void *v_t, int n_rows, int seq_len_per_head, int heads, int layout, void *stream) {
    if (n_rows <= 0 || seq_len_per_head <= 0 || (n_rows % seq_len_per_head))
        return 2;
    const int bh = n_rows / seq_len_per_head;
    const int num_kv_blocks = (seq_len_per_head + KvBlockRows - 1) / KvBlockRows;
    const int iters = bh * num_kv_blocks;
    // heads/layout select the source V layout: 0 = BHSD ([B,H,S,D]); 1 = BSHD ([B,S,H,D], strided).
    if (layout)
        hipLaunchKernelGGL(v_transpose_kernel_16x16<true>, dim3(iters), dim3(XposeThreadsPerWG), 0, (hipStream_t)stream, (const bf16_t *)v, (bf16_t *)v_t, seq_len_per_head, heads);
    else
        hipLaunchKernelGGL(v_transpose_kernel_16x16<false>, dim3(iters), dim3(XposeThreadsPerWG), 0, (hipStream_t)stream, (const bf16_t *)v, (bf16_t *)v_t, seq_len_per_head,
                           heads);
    return (int)hipGetLastError();
}
}
