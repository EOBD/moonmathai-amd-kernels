// gfx950 bf16 attention forward, head dim 128: strided, varlen (THD) or paged K/V, causal, GQA and optional LSE.
// C API: attention.h; work lists and KV split: attention_plan.hpp; notation and [tag] rules: attention_prims.hpp.
// A CTA: 384 q rows of one sequence and KV head; 8 waves of 48 rows (three 16-row q tiles) or 32 (two).
// Per KV tile a wave runs a BLOCK (S(t) = Q K(t)^T, O += P(t-1) V(t-1)) then a SOFTMAX (P(t) = exp2(S*scale - frame)).
// Waves w and w + 4 share a SIMD; group A runs half a tile ahead of B; one barrier (|) per tile:
//   A:  block 0 | softmax 0 | block 1 | softmax 1 | ...
//   B:          | block 0 | softmax 0 | block 1 | ...
// S and O are transposed (q on the MFMA N axis); the frame is a lazily rescaled row max, applied through QK's C.

#include "attention.h"
#include "attention_plan.hpp"

namespace attn {

constexpr int HeadDim = 128;
constexpr int DKSteps = HeadDim / 32;              // QK k-steps
constexpr int DMTiles = HeadDim / 16;              // PV M-tiles (16 d each)

// LDS: 3-deep K and V rings (tile t in buffer t % 3) and one Q slot per wave. A tile is 16 slabs of 512
// elements; wave w's DMA fills 2 of them, rows 8w .. 8w + 7 (one page when paged).
//   K tile [d/64][kv/8][d%64/16][kv%8][d%16/8 ^ kv%8/4][d%8]    V tile [kv/4][d/16][kv%4][d%16]
//   Q slot [q][d/8%8][d/64][d%8]
// Pitches and the K swizzle keep each 16-lane read pass on distinct 16-B chunks mod 256 B, so no bank conflicts.
constexpr int SlabsPerTile    = KvTile * HeadDim / SlabElems;
constexpr int DmasPerTile = SlabsPerTile / NumWaves;
constexpr int BlockDmas = 2 * DmasPerTile;     // K and V DMA instructions per wave and block
constexpr int KPitch = SlabElems + 32, KHalf = 8 * KPitch;         // K slab pitch; one half sits at d/64 = 1
constexpr int VPitch = SlabElems + 64;
constexpr int TileBufs = 3;
constexpr int KBufElems = 2 * KHalf, VBufElems = SlabsPerTile * VPitch;
constexpr int KRegion = TileBufs * KBufElems, VRegion = TileBufs * VBufElems;
// Tile t's buffer offset within its region; unsigned, so no signed-divide sequence.
constexpr int k_buf(int t) { return (unsigned)t % TileBufs * KBufElems; }
constexpr int v_buf(int t) { return (unsigned)t % TileBufs * VBufElems; }
constexpr int QPitch = HeadDim + 16, QSlotElems = 16 * QPitch;   // padded: no bank conflicts
constexpr int LdsElems = KRegion + VRegion + NumWaves * QSlotElems;
static_assert(LdsElems * (int)sizeof(bf16_t) <= 163840, "gfx950 allows 160 KB");
constexpr auto KTileStride =
    opus::make_tuple(opus::number<KHalf>{}, opus::number<KPitch>{}, 128_I, 16_I, 8_I, 1_I);
constexpr auto VTileStride = opus::make_tuple(opus::number<VPitch>{}, 64_I, 16_I, 1_I);
constexpr auto QSlotStride = opus::make_tuple(opus::number<QPitch>{}, 16_I, 8_I, 1_I);
// Slot of a K row's 8-element chunk c within d%16 (the swizzle).
__device__ constexpr int k_chunk(int c, int kv) { return c ^ (kv % 8 / 4); }
constexpr int OPitch = HeadDim + 8;                // O staging, in the K/V region
static_assert(NumWaves * MaxWaveRows * OPitch <= LdsElems, "O staging must fit");

// Block t: 16 PV steps (ks, dt), then 16 QK steps (ks, kt), one V^T or K fragment each (a QK kt = 0 step also
// reads q tile 2) and an MFMA per q tile.
constexpr int PvSteps = KvKSteps * DMTiles, BlockSteps = PvSteps + DKSteps * KvMTiles;
struct BlockStep {
    bool pv;
    int ks, dt, kt;
    int reads;                                     // LDS read instructions
};
template <int QTiles>
constexpr BlockStep block_step(int j)
{
    if (j < PvSteps) return {true, j / DMTiles, j % DMTiles, 0, 2};
    const int ks = (j - PvSteps) / KvMTiles, kt = (j - PvSteps) % KvMTiles;
    return {false, ks, 0, kt, kt == 0 && QTiles == 3 ? 2 : 1};
}

// QTiles: 16-row q tiles per wave (3 or 2). Split: the launch writes LSEs or split-q partials. Paged: K/V come
// from pages.
template <int QTiles, bool Split, bool Paged = false>
__global__ __launch_bounds__(BlockSize, 1) void fwd(Args a)
{
    static_assert(QTiles == 2 || QTiles == 3);
    constexpr int WaveRows = 16 * QTiles;
    constexpr int LaneScores = KvMTiles * 4 * QTiles;   // S (and P) values per lane
    constexpr int ReadLead = QTiles == 3 ? 2 : 3;     // steps between a fragment's LDS read and its MFMA
    __shared__ bf16_t lds[LdsElems];
    const int lane = threadIdx.x & (WaveSize - 1);
    const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x / WaveSize);   // lives in an SGPR
    const int g = lane >> 4, l = lane & 15;
    const bool group_a = wave < NumWaves / 2;

    // ==== work ====================================================================================
    // This wave's q rows (from work.row0) and its KV tiles.
    const WaveWork work = wave_work<Split>(a, wave, WaveRows);
    if (work.sq == 0) return;                      // an empty item of a device-built varlen list

    // Q loads, staged after the K/V DMA starts. Rows past the sequence load row 0 and stage zeros.
    constexpr int RowVecs = HeadDim / 8, QTileLoads = 16 * RowVecs / WaveSize;
    auto q_row = [&](int qt, int i) INLINE {
        return work.row0 + 16 * qt + (i * WaveSize + lane) / RowVecs;
    };
    const bf16_t* const q_rows = qo_rows(a.q, a.q_str, work);
    bf16_t* const o_rows = qo_rows(a.o, a.o_str, work);
    pin_scalar(o_rows);                            // computed before the loop
    __shared__ LseRows lse_stash[NumWaves];        // parked in LDS: the loop needs every SGPR
    if constexpr (Split) {
        const LseRows lse = lse_rows(a, work);
        if (lane == 0) lse_stash[wave] = lse;
    }
    bf16x8_t q_load[QTiles][QTileLoads];
    for (int qt = 0; qt < QTiles; ++qt)
        for (int i = 0; i < QTileLoads; ++i) {
            const int q = q_row(qt, i);
            const bf16_t* row = row_at(q_rows, a.q_str, (unsigned)q < (unsigned)work.sq ? q : 0);
            q_load[qt][i] = *reinterpret_cast<const bf16x8_t*>(row + 8 * (lane % RowVecs));
        }

    // ==== K/V DMA =================================================================================
// DMA instruction i of wave w fills K slab (d/64 = i, kv/8 = w) and V slab (kv/4 = 2w + i); lane L writes
// 16 B at 16L from rows 8w .. 8w + 7:
    //   K [kv][d] at [L%16/2][i, L/16, k_chunk(L%2, L%16/2), 8]    V [kv][d] at [i, L%8/2][L/8, L%2, 8]
    // Token strides are at most 8 MB (host::valid_strides).
    const uint32_t k_row_bytes = a.k_str.token * sizeof(bf16_t), v_row_bytes = a.v_str.token * sizeof(bf16_t);
    const auto k_src = lane_view<xdim<P<8>>, xdim<Y<2>, P<4>, P<2>, Y<8>>>(
        opus::make_tuple(k_row_bytes, 2_I), lane % 16 / 2, lane / 16, k_chunk(lane % 2, lane % 16 / 2));
    const auto v_src = lane_view<xdim<Y<2>, P<4>>, xdim<P<8>, P<2>, Y<8>>>(
        opus::make_tuple(v_row_bytes, 2_I), lane % 8 / 2, lane / 8, lane % 2);

    // Buffer resources span tile rsrc_t0 through the sequence's last key: rows past it read 0, not the next
    // sequence's (0 x NaN V is NaN). The tile offset is in soffset, which is range-checked, so resources advance
// every rsrc_tiles tiles (paged: each block builds the next one's).
    const bf16_t* const k_rows = kv_rows(a.k, a.k_str, work);
    const bf16_t* const v_rows = kv_rows(a.v, a.v_str, work);
    const size_t num_rows = work.kv_len - work.tile_lo * KvTile;   // counted from kv_row0
    auto kv_rsrc = [&](const bf16_t* rows, uint32_t row_bytes, int t) INLINE {
        return kv_rsrc_words(rows, (size_t)t * KvTile + 8 * wave, num_rows, row_bytes,
                             HeadDim * sizeof(bf16_t));
    };
    const uint32_t k_tile_bytes = KvTile * k_row_bytes, v_tile_bytes = KvTile * v_row_bytes;
    const int rsrc_tiles = Paged ? INT_MAX / 2 : UINT32_MAX / max(k_tile_bytes, v_tile_bytes) - 2;
    int rsrc_t0 = 0;
    rsrc_t k_rsrc = kv_rsrc(k_rows, k_row_bytes, 0), v_rsrc = kv_rsrc(v_rows, v_row_bytes, 0);
    auto move_rsrc = [&](int t) INLINE {
        if (!Paged && t - rsrc_t0 >= rsrc_tiles) {
            rsrc_t0 = t;
            k_rsrc = kv_rsrc(k_rows, k_row_bytes, t);
            v_rsrc = kv_rsrc(v_rows, v_row_bytes, t);
        }
    };

// Paged: rows 8w .. 8w + 7 of a tile lie in one page (8 divides the page size). A resource starts at that page
// (soffset 0) and spans up to 256 rows, capped at the last key; V's is the previous block's K resource shifted
// by v - k (shared strides). k_page_rsrc(i): piece i of the next K resource (rows k_vk + 64, page k_page) and
// of the page-after-next's table entry.
    const uint64_t k_pool = (uint64_t)(a.k + offset(a.k_str, 0, 0, work.kv_head));
    const uint64_t v_delta = (uint64_t)a.v - (uint64_t)a.k;
    const uint32_t page_bytes = a.k_str.batch * 2, page_mask = (1u << a.page_shift) - 1;
    const uint32_t kv_end = work.kv_row0 + num_rows;   // in virtual tokens
    const uint32_t kv_last = work.kv_len > 0 ? kv_end - 1 : 0;
    uint32_t k_vk = work.kv_row0 + 8 * wave - KvTile, k_row, k_entry;
    int k_page, k_next_page, k_left;               // k_next_page is in flight
    uint64_t k_base;
    auto k_page_rsrc = [&](int i) INLINE {
        switch (i) {
        case 0: anchor(k_page), k_base = (uint64_t)(uint32_t)k_page * page_bytes, anchor(k_base); break;
        case 1: k_vk += KvTile, k_left = (int)(kv_end - k_vk), anchor(k_vk), anchor(k_left); break;
        case 2: k_row = k_vk & page_mask, k_left = max(k_left, 0), anchor(k_row), anchor(k_left); break;
        case 3: k_row *= k_row_bytes, k_left = min(k_left, 256), anchor(k_row), anchor(k_left); break;
        case 4: k_base += k_row, anchor(k_base); break;
        case 5: k_base += k_pool, k_rsrc.x = k_base, k_rsrc.y = k_base >> 32, anchor(k_rsrc); break;
        case 6: k_rsrc.z = k_left * k_row_bytes, k_entry = k_vk + 2 * KvTile;
                anchor(k_rsrc), anchor(k_entry); break;
        case 7: k_entry = min(k_entry, kv_last) >> a.page_shift, anchor(k_entry); break;
        case 8: k_entry <<= 2, anchor(k_entry); break;
        }
    };
    constexpr int KPagePieces = 9;
    auto v_base = [&]() INLINE { return ((uint64_t)k_rsrc.y << 32 | k_rsrc.x) + v_delta; };
    // Take the next block's page (loaded a tile earlier), then prefetch the one after it.
    auto next_page = [&]() INLINE {
        wait_page(k_next_page);
        k_page = k_next_page;
        k_next_page = load_page(a.page_table, k_entry);
    };

    auto k_slab = [&](int t, int i) INLINE {
        return lds + k_buf(t) + i * KHalf + wave * KPitch;
    };
    auto v_slab = [&](int t, int i) INLINE {
        return lds + KRegion + v_buf(t) + (2 * wave + i) * VPitch;
    };
    auto load_k = [&](int t, uint32_t off, int i) INLINE {
        buffer_load_dwordx4_lds(k_rsrc, (lds_ptr_t)k_slab(t, i), k_src(i), off);
    };
    auto load_v = [&](int t, uint32_t off, int i) INLINE {
        buffer_load_dwordx4_lds(v_rsrc, (lds_ptr_t)v_slab(t, i), v_src(i), off);
    };

    // Block t loads K(t + k_ahead) and V(t + v_ahead); the buffers are unread until the next barrier (A's
    // block 0 reloads V(0), unchanged).
    const int k_ahead = group_a ? 1 : 2, v_ahead = group_a ? 0 : 1;

    // Block t's DMA registers, set in SALU before the block [salu]: soffsets and slab addresses (M0).
    uint32_t k_off = 0, v_off = 0, dma_lds[BlockDmas];
    auto set_dma = [&](int t) INLINE {
        if constexpr (!Paged) {
            k_off = (t - rsrc_t0 + k_ahead) * k_tile_bytes;
            v_off = (t - rsrc_t0 + v_ahead) * v_tile_bytes;
        }
        for (int i = 0; i < DmasPerTile; ++i) {
            dma_lds[i] = (uint32_t)(uintptr_t)(lds_ptr_t)k_slab(t + k_ahead, i);
            dma_lds[DmasPerTile + i] = (uint32_t)(uintptr_t)(lds_ptr_t)v_slab(t + v_ahead, i);
        }
    };

    // Paged: after its DMA, block t builds block t + 1's resources, one piece per step (V's 2, then K's) [salu].
    constexpr int NextRsrcPieces = 2 + KPagePieces;
    auto next_rsrc = [&](int i) INLINE {
        if (i == 0) v_rsrc.x = v_base(), v_rsrc.y = v_base() >> 32, anchor(v_rsrc);
        if (i == 1) v_rsrc.z = k_rsrc.z, anchor(v_rsrc);
        if (i >= 2) k_page_rsrc(i - 2);
    };

    // Block step j's DMA [dma]: instruction n's M0 at step 4n (DMA 0's in prep_block), its load at 4n + 1,
    // ordered after the step's fragment.
    auto dma_step = [&](int j, const bf16x8_t& frag) INLINE {
        const int n = j / 4, i = j - (4 * BlockDmas - 2);   // i: the paged piece
        if (j % 4 == 0 && n > 0 && n < BlockDmas) set_m0(dma_lds[n]);
        if (j % 4 == 1 && n < BlockDmas) {
            if (n < DmasPerTile) buffer_load_dwordx4_lds_m0(k_rsrc, k_src(n), k_off, frag);
            else buffer_load_dwordx4_lds_m0(v_rsrc, v_src(n - DmasPerTile), v_off, frag);
        }
        if (Paged && i >= 0 && i < NextRsrcPieces) next_rsrc(i);
    };

    // Prologue: K(0), V(0) and B's K(1) (A's comes in block 0). Paged: pages of tiles 0 .. 4 are loaded and
    // waited for together [lgkm]; each K resource is built in full, block 0's last.
    int pg[5];
    if constexpr (Paged) {
        for (int t = 0; t < 5; ++t)
            pg[t] = load_page(a.page_table, page_entry(k_vk + (t + 1) * KvTile, kv_last, a.page_shift));
        for (int t = 0; t < 5; ++t) wait_page(pg[t]);
    }
    auto k_rsrc_at = [&](int page) INLINE {
        k_page = page;
        for (int i = 0; i < KPagePieces; ++i) k_page_rsrc(i);
    };
    auto v_rsrc_at_k = [&]() INLINE {
        v_rsrc = rsrc_t{(uint32_t)v_base(), (uint32_t)(v_base() >> 32), k_rsrc.z, k_rsrc.w};
    };
    if constexpr (Paged) k_rsrc_at(pg[0]), v_rsrc_at_k();
    for (int i = 0; i < DmasPerTile; ++i) load_k(0, 0, i);
    for (int i = 0; i < DmasPerTile; ++i) load_v(0, 0, i);
    if (!group_a) {
        if constexpr (Paged) k_rsrc_at(pg[1]);
        for (int i = 0; i < DmasPerTile; ++i) load_k(1, Paged ? 0 : k_tile_bytes, i);
    }
    if constexpr (Paged) {
        v_rsrc_at_k();
        k_vk = work.kv_row0 + 8 * wave + (k_ahead - 1) * KvTile;
        k_rsrc_at(group_a ? pg[1] : pg[2]);
        k_page = group_a ? pg[2] : pg[3], k_next_page = group_a ? pg[3] : pg[4];
    }

    // ==== Q =======================================================================================
    // Q passes through the wave's LDS slot into QK's B layout. Tiles 0 and 1 stay in registers; tile 2 is
    // re-read from the slot during QK.
    const auto q_store = lane_view<xdim<Y<4>, P<4>>, xdim<P<8>>, xdim<P<2>>, xdim<Y<8>>>(
        QSlotStride, lane / 16, lane % 8, lane / 8 % 2);
    const auto q_read = lane_view<xdim<P<16>>, xdim<Y<4>, P<2>>, xdim<P<2>>, xdim<Y<8>>>(
        QSlotStride, l, g / 2, g % 2);
    bf16_t* const q_slot = lds + KRegion + VRegion + wave * QSlotElems;
    lds_addr_t q_addr = (lds_addr_t)(q_slot + q_read.lane);
    pin_after(q_addr);                             // one register, no add per block
    bf16x8_t q_frag[QTiles][DKSteps];              // QK's B operands
    for (int qt = 0; qt < QTiles; ++qt) {
        for (int i = 0; i < QTileLoads; ++i) {
            const bool valid = (unsigned)q_row(qt, i) < (unsigned)work.sq;
            *reinterpret_cast<bf16x8_t*>(q_slot + q_store(i)) = valid ? q_load[qt][i] : bf16x8_t{};
        }
        if (qt < 2)
            for (int ks = 0; ks < DKSteps; ++ks) q_frag[qt][ks] = ds_read_b128(q_addr + q_read.step(ks));
    }

    // ==== state ===================================================================================
    f32x4_t o_acc[DMTiles][QTiles] = {};           // O^T(dt, qt)[i] = O^T[d = 16dt + 4g + i][q = 16qt + l]

    // S^T(kt, qt)[i] = S^T[kv][q = 16qt + l], with kv at [kt/2, g/2, kt%2, g%2, i] (C row 4g + i of K(kt)).
    f32x4_t s_acc[KvMTiles][QTiles];
    const float scale = a.scale_log2e;
    const float threshold = readfirstlane(RescaleThreshold / scale);   // in S units; SGPR
    const float inv_scale = readfirstlane(1.0f / scale);

    // QK's C operand, -frame / scale per row, so S arrives relative to the frame.
    f32x4_t s_seed[QTiles];
    for (int qt = 0; qt < QTiles; ++qt) s_seed[qt] = -Frame0 / scale;

    // PV's B: p_frag[ks][qt] holds S tiles 2ks and 2ks + 1 in V^T's kv order.
    bf16x8_t p_frag[KvKSteps][QTiles] = {};

    // Row sums of P by sparse MFMA: B stacks two P fragments along k (the first at k < 32); A is 2:4-sparse ones,
    // row m keeping fragment m % 4 / 2 at k positions 0, 1 (even m) or 2, 3 (odd m) of each four.
    // C0 + C1 and C2 + C3 are the two fragments' row sums.
    const bf16x8_t ones = (bf16_t)(((l & 3) < 2) == (g < 2) ? 1.0f : 0.0f);   // row l, k 16g .. 16g + 15
    const int ones_idx = (int)((l & 1) ? 0xEEEEEEEEu : 0x44444444u);
    f32x4_t row_sum01{}, row_sum2{};               // q tiles 0 and 1; q tile 2's pair of k-steps
    auto row_sums = [&]() INLINE {
        row_sum01 = v_smfmac_f32_16x16x64_bf16(ones, join(p_frag[0][0], p_frag[0][1]), row_sum01, ones_idx);
        row_sum01 = v_smfmac_f32_16x16x64_bf16(ones, join(p_frag[1][0], p_frag[1][1]), row_sum01, ones_idx);
        if constexpr (QTiles == 3)
            row_sum2 = v_smfmac_f32_16x16x64_bf16(ones, join(p_frag[0][2], p_frag[1][2]), row_sum2, ones_idx);
    };

    // ==== LDS reads ===============================================================================
    // K fragment (kt, ks), QK's A: row l, columns 8g .. 8g + 7, in the K tile at
    // [g%2][kt/2, l/8, kt%2][ks][l%8][k_chunk(g/2, l)][8].
    const auto k_read =
        lane_view<xdim<P<2>>, xdim<Y<2>, P<2>, Y<2>>, xdim<Y<4>>, xdim<P<8>>, xdim<P<2>>, xdim<Y<8>>>(
            KTileStride, g % 2, l / 8, l % 8, k_chunk(g / 2, l));
    // V^T fragment (ks, dt), PV's A: two ds_read_b64_tr_b16 (h = 0, 1) at [ks, g/2, h, g%2][dt][l/4][l%4, 4]
    // of the V tile; lane l receives column d = 16dt + l.
    const auto v_read = lane_view<xdim<Y<2>, P<2>, Y<2>, P<2>>, xdim<Y<8>>, xdim<P<4>>, xdim<P<4>, Y<4>>>(
        VTileStride, g / 2, g % 2, l / 4, l % 4);

    // Step s's fragment (QK kt = 0 also reads q tile 2).
    auto read_step = [&](lds_addr_t v_addr, lds_addr_t k_addr, BlockStep s) INLINE {
        if (s.pv)
            return join(ds_read_b64_tr_b16(v_addr + v_read.step(s.ks, 0, s.dt)),
                        ds_read_b64_tr_b16(v_addr + v_read.step(s.ks, 1, s.dt)));
        if constexpr (QTiles == 3)
            if (s.kt == 0) q_frag[2][s.ks] = ds_read_b128(q_addr + q_read.step(s.ks));
        return ds_read_b128(k_addr + k_read.step(s.kt / 2, s.kt % 2, s.ks));
    };

    // ==== next block's set-up =====================================================================
    // Prepared in softmax t - 1 so block t starts with an MFMA: DMA registers, read addresses and first fragments.
    lds_addr_t v_lane = (lds_addr_t)(lds + KRegion + v_read.lane);   // each tile's v_addr is one add
    pin_after(v_lane);
    int block_v_buf, block_k_buf, next_v_buf;      // V(t - 1), K(t), V(t) (B's look-ahead)
    lds_addr_t v_addr, k_addr, v_next;
    bf16x8_t first_frags[ReadLead];
    auto prep_block = [&](int t) INLINE {
        set_dma(t);
        set_m0(dma_lds[0]);                        // DMA 0's M0; the block sets the others
        block_v_buf = v_buf(max(t - 1, 0));        // block 0 (P = 0) reads V(0): a stale one may hold NaNs
        block_k_buf = k_buf(t);
        next_v_buf = v_buf(t);
    };
    auto prep_block_addrs = [&]() INLINE {         // pinned: computed here, not in the block
        v_addr = v_lane + block_v_buf;
        k_addr = (lds_addr_t)(lds + block_k_buf + k_read.lane);
        v_next = v_lane + next_v_buf;
        pin_after(v_addr);
        pin_after(k_addr);
    };
    auto prep_block_frags = [&]() INLINE {
        for (int i = 0; i < ReadLead; ++i) first_frags[i] = read_step(v_addr, k_addr, block_step<QTiles>(i));
    };

    // ==== frames ==================================================================================
    // q tile qt's largest S in this lane, as a max3 tree. keep_nan: NaN if any S is NaN (for the frame test).
    // MaskScore is a 17th leaf so every node takes three operands.
    auto lane_max = [&](int qt, auto keep_nan) INLINE {
        auto max = [](float a, float b, float c) INLINE {
            if constexpr (decltype(keep_nan)::value) return maybe_nan(maximum3(a, b, c));
            else return max3(a, b, c);
        };
        float s[KvMTiles * 4];
        for (int kt = 0; kt < KvMTiles; ++kt)
            for (int i = 0; i < 4; ++i) s[4 * kt + i] = s_acc[kt][qt][i];
        const float m0 = max(s[0], s[1], s[2]), m1 = max(s[3], s[4], s[5]), m2 = max(s[6], s[7], s[8]);
        const float m3 = max(s[9], s[10], s[11]), m4 = max(s[12], s[13], s[14]);
        return max(MaskScore, max(m0, m1, m2), max(m3, m4, s[15]));
    };
    // Row maxima of every q tile, over the row's 4 lanes.
    auto row_maxes = [&](float (&m)[QTiles]) INLINE {
        for (int qt = 0; qt < QTiles; ++qt) m[qt] = lane_max(qt, std::false_type{});
        max_lane_xor32(m);
        max_lane_xor16(m);
    };

    // Frame updates of q tile qt in scalar fp32 [pk]; shift in S units.
    auto shift_frames = [&](int qt, float shift) INLINE {
        for (int i = 0; i < 4; ++i) s_seed[qt][i] -= shift;
        for (int kt = 0; kt < KvMTiles; ++kt)
            for (int i = 0; i < 4; ++i) s_acc[kt][qt][i] -= shift;
    };
    auto scale_o = [&](int qt, float r) INLINE {
        for (int dt = 0; dt < DMTiles; ++dt)
            for (int i = 0; i < 4; ++i) o_acc[dt][qt][i] *= r;
    };
    auto scale_row_sums = [&](int qt, float r) INLINE {
        if (qt < 2) {
            row_sum01[2 * qt] *= r;
            row_sum01[2 * qt + 1] *= r;
        } else {
            for (int i = 0; i < 4; ++i) row_sum2[i] *= r;
        }
    };

    // First tile: if any row's frame is more than Frame0 above its max (P would underflow), every row's frame
    // moves to Frame0 above its own max.
    auto set_row_frames = [&]() INLINE {
        bool low = false;
        for (int qt = 0; qt < QTiles; ++qt) {
            const float m = lane_max(qt, std::false_type{});
            low |= (m > NegInf) & (m * scale < -Frame0);   // &: no per-lane branch
        }
        if (!any_lane(low)) return;
        float m[QTiles];
        row_maxes(m);
        for (int qt = 0; qt < QTiles; ++qt)                 // seed becomes -Frame0 / scale
            shift_frames(qt, m[qt] > NegInf ? m[qt] - s_seed[qt][0] : 0.0f);
    };

    // Later tiles: once a row max exceeds its frame by RescaleThreshold, every row's frame moves to
    // RescaleHeadroom above its max. S and A's row sums update now; O, seeds and B's row sums update at the next
    // block's start. A NaN or +inf S takes this path too.
    bool frames_moved = false;                     // set by the latest softmax
    float m[QTiles];                               // the test's lane maxima, then the shifts
    auto frames_outgrown = [&]() INLINE {
        for (int qt = 0; qt < QTiles; ++qt) m[qt] = lane_max(qt, std::true_type{});   // in S units
        return lanes_above_or_nan(maybe_nan(maximum_n(m)), threshold) != 0;
    };
    // A row with a NaN or +inf S: its S become MaskScore (a sparse MFMA 0 x NaN would reach other rows), its
    // frame stays, its row sum becomes NaN (so its O is NaN). bad: the lanes holding such an S, per q tile.
    auto mask_nonfinite = [&](const uint64_t (&bad)[QTiles]) INLINE {
        const int l = fresh_lane_id() & 15;        // not live outside this rare path
        const float nan = opaque_nan();
        for (int qt = 0; qt < QTiles; ++qt) {
            const bool row_bad = has_row(bad[qt], l);
            for (int kt = 0; kt < KvMTiles; ++kt)
                for (int i = 0; i < 4; ++i) s_acc[kt][qt][i] = row_bad ? MaskScore : s_acc[kt][qt][i];
            m[qt] = row_bad ? 0.0f : m[qt];
            scale_row_sums(qt, row_bad ? nan : 1.0f);
        }
    };
    auto move_frames = [&](auto group_a) INLINE {
        uint64_t bad[QTiles];                      // lanes holding a NaN or +inf S
        for (int qt = 0; qt < QTiles; ++qt) bad[qt] = lanes_above_or_nan(m[qt], MaxFinite);
        if constexpr (!group_a) s_setprio<1>();    // B's softmax is the long path here
        max_lane_xor32(m);
        max_lane_xor16(m);
        for (int qt = 0; qt < QTiles; ++qt) {
            m[qt] = fmaxf(m[qt] * scale + RescaleHeadroom, 0.0f);   // frames only ever move up
            for (int kt = 0; kt < KvMTiles; ++kt)
                for (int i = 0; i < 4; ++i) s_acc[kt][qt][i] -= m[qt];
            if constexpr (group_a) scale_row_sums(qt, __builtin_amdgcn_exp2f(-m[qt]));
        }
        if constexpr (!group_a) s_setprio<0>();
        sched_fence();                             // the compares' SALU last [vcc]
        uint64_t any_bad = 0;
        for (int qt = 0; qt < QTiles; ++qt) any_bad |= bad[qt];
        if (unlikely(any_bad != 0)) mask_nonfinite(bad);
    };
    auto follow_frames = [&](auto group_a) INLINE {
        for (int qt = 0; qt < QTiles; ++qt) {
            const float r = __builtin_amdgcn_exp2f(-m[qt]);
            for (int i = 0; i < 4; ++i) s_seed[qt][i] -= m[qt] * inv_scale;
            if constexpr (!group_a) scale_row_sums(qt, r);
            scale_o(qt, r);
        }
    };

    // ==== softmax =================================================================================
    // Keys past the sequence or past a causal row's last key: S = MaskScore.
    auto mask_scores = [&](int kv0) INLINE {
        const int lane = fresh_lane_id(), g = lane >> 4, l = lane & 15;   // not live outside this rare path
        // kv of this lane's S value (kt, i).
        const auto s_kv = lane_view<xdim<Y<2>, P<2>, Y<2>, P<2>, Y<4>>>(opus::make_tuple(1_I), g / 2, g % 2);
        for (int qt = 0; qt < QTiles; ++qt) {
            int kv_limit = work.kv_len - 1;
            if (a.causal) kv_limit = min(kv_limit, work.row0 + 16 * qt + l + work.kv_shift);
            kv_limit -= kv0 + s_kv.lane;
            for (int kt = 0; kt < KvMTiles; ++kt)
                for (int i = 0; i < 4; ++i)
                    s_acc[kt][qt][i] = s_kv.step(kt / 2, kt % 2, i) > kv_limit ? MaskScore : s_acc[kt][qt][i];
        }
    };

    // Rare tiles: the first, the masked ones (through the end), and the last before the resources advance; one
    // test per tile [branch].
    int next_rare_tile = 0;                            // the next rare tile
    auto tile_extras = [&](int t) INLINE {
        if (t >= work.num_unmasked) mask_scores((work.tile_lo + t) * KvTile);
        if (t == 0) set_row_frames();
        move_rsrc(t + 1);
        next_rare_tile = max(t + 1, min(work.num_unmasked, rsrc_t0 + rsrc_tiles - 1));
    };

    // Per-score passes in scalar fp32 [pk].
    auto scale_scores = [&]() INLINE {
        for (int kt = 0; kt < KvMTiles; ++kt)
            for (int qt = 0; qt < QTiles; ++qt)
                for (int i = 0; i < 4; ++i) s_acc[kt][qt][i] *= scale;
    };
    auto exp_scores = [&]() INLINE {
        for (int kt = 0; kt < KvMTiles; ++kt)
            for (int qt = 0; qt < QTiles; ++qt)
                for (int i = 0; i < 4; ++i) s_acc[kt][qt][i] = __builtin_amdgcn_exp2f(s_acc[kt][qt][i]);
    };
    auto convert_p = [&]() INLINE {
        for (int ks = 0; ks < KvKSteps; ++ks)
            for (int qt = 0; qt < QTiles; ++qt)
                for (int s = 0; s < 8; ++s) p_frag[ks][qt][s] = (bf16_t)s_acc[2 * ks + s / 4][qt][s % 4];
    };

    // Softmax t: P(t) = exp2(S scale - frame) in three phases; rare work out of line [branch].
    auto softmax = [&](auto group_a, int t) INLINE {
        if constexpr (Paged && group_a) next_page();   // B's later: its first reads are in flight [lgkm]
        if (unlikely(t >= next_rare_tile)) tile_extras(t);

        // 1. S to log2 units; if any row outgrew its frame, all frames move.
        frames_moved = frames_outgrown();
        scale_scores();                            // covers the test's latency
        __builtin_amdgcn_sched_group_barrier(SchedValu, 2 * LaneScores, 0);   // all VALU (< 2 per score),
        __builtin_amdgcn_sched_group_barrier(SchedSalu, 2, 0);                // then the test's SALU [vcc]
        if (unlikely(frames_moved)) move_frames(group_a);

        // 2. Exps back to back [mfma], the next block's SALU set-up between them [salu].
        if constexpr (Paged && !group_a) next_page();
        exp_scores();
        prep_block(t + 1);
        for (int i = 0; i < LaneScores; ++i) {     // exp, SALU, exp, SALU, ...
            __builtin_amdgcn_sched_group_barrier(SchedTrans, 1, 0);
            __builtin_amdgcn_sched_group_barrier(SchedSalu, 1, 0);
        }
        sched_fence();

        // 3. P to bf16; the next block's addresses (A: also its first fragments and P's row sums).
        convert_p();
        prep_block_addrs();
        if constexpr (group_a) {                   // B reads its first fragments in its block
            pin_before(p_frag);                    // reads after the conversions [lds]
            prep_block_frags();
            sched_fence();
            row_sums();                            // covers the loop's back edge
        }
    };

    // ==== the MFMA block ==========================================================================
    auto mfma_step = [&](BlockStep s, bf16x8_t frag) INLINE {
        for (int qt = 0; qt < QTiles; ++qt) {
            if (s.pv) {
                o_acc[s.dt][qt] = v_mfma_f32_16x16x32_bf16(frag, p_frag[s.ks][qt], o_acc[s.dt][qt]);
            } else {
                const f32x4_t c = s.ks == 0 ? s_seed[qt] : s_acc[s.kt][qt];
                s_acc[s.kt][qt] = v_mfma_f32_16x16x32_bf16(frag, q_frag[qt][s.ks], c);
            }
        }
    };

    // A step's order: MFMA, read, MFMA, read, MFMA; read i issues behind MFMA i (a step has at most 2 reads).
    auto order_step = [](int reads) INLINE {
        for (int qt = 0; qt < QTiles; ++qt) {
            __builtin_amdgcn_sched_group_barrier(SchedMfma, 1, 0);
            if (qt < reads) __builtin_amdgcn_sched_group_barrier(SchedDsRead, 1, 0);
        }
    };

    // Wait for this wave's DMA, then for all waves; fenced.
    auto tile_barrier = []() INLINE {
        s_waitcnt_vmcnt<0>();
        sched_fence();
        s_barrier();
        sched_fence();
    };

    // Block t: PV(t-1) then QK(t) steps, each reading ReadLead steps ahead; steps are fenced [fence].
    auto mfma_block = [&](auto group_a) INLINE {
        if (unlikely(frames_moved)) follow_frames(group_a);   // beside the partner's softmax
        if constexpr (!group_a) {                  // B's block first; A's MFMAs fill its gaps
            s_setprio<1>();                        // before the barrier: B's MFMAs issue on its release
            tile_barrier();
            row_sums();                            // of P(t - 1), ahead of its PV
        }

        bf16x8_t frag[BlockSteps];
        for (int i = 0; i < ReadLead; ++i) frag[i] = first_frags[i];
#pragma unroll                                     // j has to stay a constant
        for (int j = 0; j < BlockSteps; ++j) {
            const int r = j + ReadLead;
            const BlockStep step = block_step<QTiles>(j), ahead = block_step<QTiles>(r % BlockSteps);

            sched_fence();
            dma_step(j, frag[j]);
            sched_fence();

            // Past the block end, B reads the next block's first fragments; A reads nothing.
            if (r < BlockSteps) frag[r] = read_step(v_addr, k_addr, ahead);
            else if constexpr (!group_a) first_frags[r - BlockSteps] = read_step(v_next, k_addr, ahead);
            mfma_step(step, frag[j]);
            order_step(r < BlockSteps || !group_a ? ahead.reads : 0);
        }
        sched_fence();                             // MFMAs before A's DMA wait
        if constexpr (group_a) tile_barrier();
        else s_setprio<0>();
    };

    // Block of a tile masked for every row of this wave: its DMA only.
    auto dma_only_block = [&](auto group_a, int t) INLINE {
        move_rsrc(t);
        if constexpr (Paged) next_page();
        set_dma(t);                                // k_off, v_off
        if constexpr (!group_a) tile_barrier();
        for (int i = 0; i < DmasPerTile; ++i) load_k(t + k_ahead, k_off, i);
        for (int i = 0; i < DmasPerTile; ++i) load_v(t + v_ahead, v_off, i);
        if constexpr (Paged) for (int i = 0; i < NextRsrcPieces; ++i) next_rsrc(i);
        if constexpr (group_a) tile_barrier();
    };

    // ==== the main loop ===========================================================================
    // How the groups differ (timeline: file header):
    //                            A                              B
    //   prologue DMA             K(0), V(0)                     K(0), V(0), K(1)
    //   block t's DMA            K(t + 1), V(t)                 K(t + 2), V(t + 1)
    //   tile barrier             after its block                before its block
    //   block t+1's first reads  after softmax t's conversions  in block t's last two steps
    //   row sums of P(t)         end of softmax t               start of block t + 1
    //   issue priority           0                              1 in its block and its frame moves
    if (group_a) s_waitcnt_vmcnt<0>();
    else         s_waitcnt_vmcnt<DmasPerTile>();       // K(1) may still be in flight
    s_barrier();                                       // make tile 0 visible to every wave

    // One copy per group; group_a is a compile-time constant [branch].
    auto main_loop = [&](auto group_a) INLINE {
        prep_block(0);
        prep_block_addrs();
        prep_block_frags();                            // B's too: block 0 has no predecessor
        for (int t = 0; t < work.num_live; ++t) {
            mfma_block(group_a);
            softmax(group_a, t);
        }
        mfma_block(group_a);                           // the final PV (its QK goes unused)
        for (int t = work.num_live + 1; t <= work.num_tiles; ++t) dma_only_block(group_a, t);
    };
    if (group_a) main_loop(std::true_type{});
    else         main_loop(std::false_type{});

    // ==== epilogue: O = O^T / row sum =============================================================
    float row_sum[QTiles] = {row_sum01[0] + row_sum01[1], row_sum01[2] + row_sum01[3]};
    if constexpr (QTiles == 3) row_sum[2] = (row_sum2[0] + row_sum2[1]) + (row_sum2[2] + row_sum2[3]);

    if (work.slot >= 0) {                          // a part of a split q block: fp32 partials for merge_parts
        for (int qt = 0; qt < QTiles; ++qt) {
            const int q = work.row0 + 16 * qt + l;
            if ((unsigned)q >= (unsigned)work.sq) continue;
            const size_t row = (size_t)work.slot * BlockM + wave * WaveRows + 16 * qt + l;
            float* o_row = a.part_o + row * HeadDim + 4 * g;
            for (int dt = 0; dt < DMTiles; ++dt)
                *reinterpret_cast<f32x4_t*>(o_row + 16 * dt) = o_acc[dt][qt];
            if (g == 0) a.part_ml[row] = float2{-s_seed[qt][0] * scale, row_sum[qt]};
        }
        return;
    }
    if constexpr (Split) {                         // LSEs of a whole q block, one lane per row
        const LseRows lse = lse_stash[wave];
        for (int qt = 0; qt < QTiles; ++qt)
            if (g == 0) store_lse(lse, work, work.row0 + 16 * qt + l, -s_seed[qt][0] * scale, row_sum[qt]);
    }

    // Stage O^T through LDS for full-row global stores, once every wave's last DMA has landed.
    s_waitcnt_vmcnt<0>();
    s_barrier();
    bf16_t* o_stage = lds + wave * (WaveRows * OPitch);
    for (int qt = 0; qt < QTiles; ++qt) {
        const float inv = inv_row_sum(row_sum[qt], nan_or_plus_inf(maybe_nan(row_sum[qt])));
        for (int dt = 0; dt < DMTiles; ++dt) {
            bf16x4_t out;
            for (int e = 0; e < 4; ++e) out[e] = (bf16_t)(o_acc[dt][qt][e] * inv);
            *reinterpret_cast<bf16x4_t*>(o_stage + (16 * qt + l) * OPitch + 16 * dt + 4 * g) = out;
        }
    }
    constexpr int RowsPerStore = WaveSize / RowVecs;
    const int id = fresh_lane_id();                    // recomputed: avoids a spill
    for (int r = 0; r < WaveRows / RowsPerStore; ++r) {
        const int row = r * RowsPerStore + id / RowVecs;
        const int d   = id % RowVecs * 8;
        const int q   = work.row0 + row;
        if ((unsigned)q < (unsigned)work.sq)
            *reinterpret_cast<bf16x8_t*>(row_at(o_rows, a.o_str, q) + d) =
                *reinterpret_cast<const bf16x8_t*>(o_stage + row * OPitch + d);
    }
}

}  // namespace attn

// ---- the C API, declared in attention.h -------------------------------------------------------
using namespace attn;

// The kernels (attention_plan.hpp); a 32-row wave's KV tile costs 0.74 of a 48-row one, and a Split=true
// kernel's tile 1.01 (32 rows) and 1.09 (48 rows) of its Split=false one's.
static const host::Kernels kernels{{{fwd<2, false>, fwd<2, true>}, {fwd<3, false>, fwd<3, true>}}, 0.74f,
                                   {1.01f, 1.09f}};
static const host::Kernels paged_kernels{{{fwd<2, false, true>, fwd<2, true, true>},
                                          {fwd<3, false, true>, fwd<3, true, true>}}, 0.74f, {1.01f, 1.09f}};

extern "C" int cdna4_attn_fwd_bf16_strided(const void* q, const void* k, const void* v, void* o, float* lse,
                                           const int64_t* strides, int B, int H, int HKv, int Sq, int Skv,
                                           int D, int causal, float scale, hipStream_t stream)
{
    if (D != HeadDim) return -1;
    return host::api_fwd<HeadDim, HeadDim>(kernels, {q, k, v, o, lse, strides}, B, H, HKv, Sq, Skv, causal,
                                           scale, stream);
}

extern "C" int cdna4_attn_varlen_fwd_bf16_strided(const void* q, const void* k, const void* v, void* o,
                                                  float* lse, const int64_t* strides, const int* cu_seqlens_q,
                                                  const int* cu_seqlens_k, int num_seqs, int H, int HKv,
                                                  int D, int causal, float scale, hipStream_t stream)
{
    if (D != HeadDim) return -1;
    return host::api_varlen_fwd<HeadDim, HeadDim>(kernels, {q, k, v, o, lse, strides}, cu_seqlens_q,
                                                  cu_seqlens_k, num_seqs, H, HKv, causal, scale, stream);
}

extern "C" int cdna4_attn_paged_fwd_bf16_strided(const void* q, const void* k, const void* v, void* o,
                                                 float* lse, const int64_t* strides, const int* cu_seqlens_q,
                                                 const int* page_table, int table_len, const int* page_starts,
                                                 const int* kv_lens, int num_seqs, int page_size, int H,
                                                 int HKv, int D, int causal, float scale, hipStream_t stream)
{
    if (D != HeadDim) return -1;
    return host::api_paged_fwd<HeadDim, HeadDim>(paged_kernels, {q, k, v, o, lse, strides}, cu_seqlens_q,
                                                 page_table, table_len, page_starts, kv_lens, num_seqs,
                                                 page_size, H, HKv, causal, scale, stream);
}

extern "C" int cdna4_attn_paged_fwd_bf16(const void* q, const void* k, const void* v, void* o,
                                         const int* cu_seqlens_q, const int* page_table, int table_len,
                                         const int* page_starts, const int* kv_lens, int num_seqs,
                                         int page_size, int H, int HKv, int D, int causal, float scale,
                                         hipStream_t stream)
{
    const auto s = host::paged_strides(page_size, H, HKv, HeadDim, HeadDim);
    return cdna4_attn_paged_fwd_bf16_strided(q, k, v, o, nullptr, s.data(), cu_seqlens_q, page_table,
                                             table_len, page_starts, kv_lens, num_seqs, page_size, H, HKv,
                                             D, causal, scale, stream);
}

extern "C" int cdna4_attn_varlen_fwd_bf16_dev_strided(const void* q, const void* k, const void* v, void* o,
                                                      float* lse, const int64_t* strides,
                                                      const int* cu_seqlens_q, const int* cu_seqlens_k,
                                                      const int* host_cu_seqlens_q,
                                                      const int* host_cu_seqlens_k, int num_seqs,
                                                      int max_seqlen_q, int max_seqlen_k, int total_q,
                                                      int total_k, int H, int HKv, int D, int causal,
                                                      float scale, hipStream_t stream)
{
    if (D != HeadDim) return -1;
    const host::VarlenCall c{cu_seqlens_q, cu_seqlens_k, host_cu_seqlens_q, host_cu_seqlens_k, num_seqs,
                             max_seqlen_q, max_seqlen_k, total_q, total_k};
    return host::api_varlen_fwd_dev<HeadDim, HeadDim>(kernels, {q, k, v, o, lse, strides}, c, H, HKv, causal,
                                                      scale, stream);
}

extern "C" int cdna4_attn_varlen_plan(void* plan, const int* cu_seqlens_q, const int* cu_seqlens_k,
                                      const int* host_cu_seqlens_q, const int* host_cu_seqlens_k,
                                      int num_seqs, int max_seqlen_q, int max_seqlen_k, int total_q,
                                      int total_k, int H, int HKv, int D, int causal, hipStream_t stream)
{
    if (D != HeadDim) return -1;
    const host::VarlenCall c{cu_seqlens_q, cu_seqlens_k, host_cu_seqlens_q, host_cu_seqlens_k, num_seqs,
                             max_seqlen_q, max_seqlen_k, total_q, total_k};
    return host::api_varlen_plan<HeadDim, HeadDim>(plan, kernels, c, H, HKv, causal, stream);
}

extern "C" int cdna4_attn_varlen_run(const void* plan, const void* q, const void* k, const void* v, void* o,
                                     float* lse, const int64_t* strides, float scale, hipStream_t stream)
{
    return host::api_varlen_run<HeadDim, HeadDim>(plan, kernels, {q, k, v, o, lse, strides}, scale, stream);
}

// Contiguous tensors: BSHD and packed THD.
extern "C" int cdna4_attn_fwd_bf16(const void* q, const void* k, const void* v, void* o, int B, int H,
                                   int HKv, int Sq, int Skv, int D, int causal, float scale,
                                   hipStream_t stream)
{
    const auto s = host::contiguous_strides(Sq, Skv, H, HKv, HeadDim, HeadDim);
    return cdna4_attn_fwd_bf16_strided(q, k, v, o, nullptr, s.data(), B, H, HKv, Sq, Skv, D, causal, scale,
                                       stream);
}

extern "C" int cdna4_attn_varlen_fwd_bf16(const void* q, const void* k, const void* v, void* o,
                                          const int* cu_seqlens_q, const int* cu_seqlens_k, int num_seqs,
                                          int H, int HKv, int D, int causal, float scale, hipStream_t stream)
{
    const auto s = host::contiguous_strides(0, 0, H, HKv, HeadDim, HeadDim);
    return cdna4_attn_varlen_fwd_bf16_strided(q, k, v, o, nullptr, s.data(), cu_seqlens_q, cu_seqlens_k,
                                              num_seqs, H, HKv, D, causal, scale, stream);
}

extern "C" int cdna4_attn_varlen_fwd_bf16_dev(const void* q, const void* k, const void* v, void* o,
                                              const int* cu_seqlens_q, const int* cu_seqlens_k,
                                              const int* host_cu_seqlens_q, const int* host_cu_seqlens_k,
                                              int num_seqs, int max_seqlen_q, int max_seqlen_k, int total_q,
                                              int total_k, int H, int HKv, int D, int causal, float scale,
                                              hipStream_t stream)
{
    const auto s = host::contiguous_strides(0, 0, H, HKv, HeadDim, HeadDim);
    return cdna4_attn_varlen_fwd_bf16_dev_strided(q, k, v, o, nullptr, s.data(), cu_seqlens_q, cu_seqlens_k,
                                                  host_cu_seqlens_q, host_cu_seqlens_k, num_seqs,
                                                  max_seqlen_q, max_seqlen_k, total_q, total_k, H, HKv, D,
                                                  causal, scale, stream);
}

extern "C" void* cdna4_attn_plan_create(void) { return new (std::nothrow) host::VarlenPlan; }

extern "C" void cdna4_attn_plan_destroy(void* plan)
{
    if (!plan) return;
    host::release(*(host::VarlenPlan*)plan);
    delete (host::VarlenPlan*)plan;
}
