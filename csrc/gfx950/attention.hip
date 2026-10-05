// gfx950 attention forward: bf16 with head dim 128; arbitrary strides, varlen (THD) or paged K/V, dense
// or causal, GQA, optional LSE. C API in attention.h.
//
// One CTA owns 384 q rows belonging to a single sequence and KV head: 8 waves, 2 per SIMD, each holding
// 48 rows of one q head as three 16-row q tiles for v_mfma_f32_16x16x32_bf16 (or 32 rows in two tiles:
// QTiles). K and V arrive over DMA, streaming through LDS in 64-row tiles. Per KV tile t a wave executes
// an MFMA block computing S(t) = Q K(t)^T and O += P(t-1) V(t-1), followed by a softmax
// P(t) = exp2(S(t) scale - frame). Waves 0-3 (group A) run half a tile in front of waves 4-7 (group B):
// within a SIMD, one wave handles its softmax while its partner keeps the matrix core occupied (see the
// main loop). A frame sits where FA2 keeps the running max (log2 units) and shifts only after the row max
// exceeds it by RescaleThreshold (lazy rescale); QK applies it at no cost through the MFMA C operand
// (s_seed). S and O are stored transposed (q on the MFMA N axis), keeping a q row's softmax in one lane.
// Work lists and the KV split: attention_plan.hpp. Notation and the [tag]-cited rules: attention_prims.hpp.

#include "attention.h"
#include "attention_plan.hpp"

namespace attn {

constexpr int HeadDim = 128;
constexpr int DKSteps = HeadDim / 32;              // k-steps per QK
constexpr int DMTiles = HeadDim / 16;              // PV M-tile count

// LDS layout: 3-deep ring buffers for K and for V (tile t occupies buffer t % 3), plus one Q slot per
// wave. A tile buffer consists of 16 slabs of 512 elements, 2 filled by each wave's DMA. One slab holds
// 8 (K) or 4 (V) kv rows in a row, so a wave's DMA covers rows 8w .. 8w + 7 of a tile (a single page
// when paged).
//   K tile [d/64][kv/8][kv%8][d%64/8 ^ kv%8/4][d%8]    V tile [kv/4][d/16][kv%4][d%16]
//   Q slot [q][d/8%8][d/64][d%8]
// Pitches and the K swizzle render the reads conflict-free: a half-wave's 16-B chunks stay distinct mod 512 B.
constexpr int SlabsPerTile    = KvTile * HeadDim / SlabElems;
constexpr int DmaInstsPerTile = SlabsPerTile / NumWaves;
constexpr int BlockDmas = 2 * DmaInstsPerTile;     // K and V DMA instructions issued per wave, per block
constexpr int KPitch = SlabElems + 8, KHalf = 8 * KPitch + 224;    // K slabs; one half at d/64 = 1
constexpr int VPitch = SlabElems + 64;
constexpr int TileBufs = 3;
constexpr int KBufElems = 2 * KHalf, VBufElems = SlabsPerTile * VPitch;
constexpr int KRegion = TileBufs * KBufElems, VRegion = TileBufs * VBufElems;
// Tile t's buffer within its region. Unsigned cast keeps the modulo from becoming a signed divide.
constexpr int k_buf(int t) { return (unsigned)t % TileBufs * KBufElems; }
constexpr int v_buf(int t) { return (unsigned)t % TileBufs * VBufElems; }
constexpr int QPitch = HeadDim + 16, QSlotElems = 16 * QPitch;   // row padding keeps reads conflict-free
constexpr int LdsElems = KRegion + VRegion + NumWaves * QSlotElems;
static_assert(LdsElems * (int)sizeof(bf16_t) <= 163840, "gfx950 allows 160 KB");
constexpr auto KTileStride = opus::make_tuple(opus::number<KHalf>{}, opus::number<KPitch>{}, 64_I, 8_I, 1_I);
constexpr auto VTileStride = opus::make_tuple(opus::number<VPitch>{}, 64_I, 16_I, 1_I);
constexpr auto QSlotStride = opus::make_tuple(opus::number<QPitch>{}, 16_I, 8_I, 1_I);
// Position of a K row's 8-element chunk c within d%64 (the swizzle).
__device__ constexpr int k_chunk(int c, int kv) { return c ^ (kv % 8 / 4); }
constexpr int OPitch = HeadDim + 8;                // O staging (borrows the K/V region)
static_assert(NumWaves * MaxWaveRows * OPitch <= LdsElems, "O staging must fit");

// An MFMA block spans 32 steps: 16 PV steps (ks, dt) then 16 QK steps (ks, kt). Each step fetches one
// V^T or K fragment (QK steps where kt = 0 additionally fetch q tile 2) and drives the wave's 3 q tiles.
constexpr int PvSteps = KvKSteps * DMTiles, BlockSteps = PvSteps + DKSteps * KvMTiles;
struct BlockStep {
    bool pv;
    int ks, dt, kt;
    int reads;                                     // number of LDS read instructions
};
template <int QTiles>
constexpr BlockStep block_step(int j)
{
    if (j < PvSteps) return {true, j / DMTiles, j % DMTiles, 0, 2};
    const int ks = (j - PvSteps) / KvMTiles, kt = (j - PvSteps) % KvMTiles;
    return {false, ks, 0, kt, kt == 0 && QTiles == 3 ? 2 : 1};
}

// QTiles: q tiles of 16 rows per wave, either 3 or 2 (attention_plan.hpp). Split: the launch contains
// split q blocks whose parts save fp32 partials, or it stores LSEs (otherwise only whole q blocks' O).
// Paged: K/V arrive as pages.
template <int QTiles, bool Split, bool Paged = false>
__global__ __launch_bounds__(BlockSize, 1) void fwd(Args a)
{
    static_assert(QTiles == 2 || QTiles == 3);
    constexpr int WaveRows = 16 * QTiles;          // rows handled per wave in this launch
    constexpr int LaneScores = KvMTiles * 4 * QTiles;   // S (and P) values held per lane
    constexpr int Ahead = QTiles == 3 ? 2 : 3;     // MFMA block read-ahead measured in steps
    __shared__ bf16_t lds[LdsElems];
    const int lane = threadIdx.x & (WaveSize - 1);
    const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x / WaveSize);   // lives in an SGPR
    const int g = lane >> 4, l = lane & 15;
    const bool group_a = wave < NumWaves / 2;

    // ==== work assignment =========================================================================
    // The WaveRows q rows of one head this wave handles, starting at work.row0, plus its KV tiles
    // (attention_plan.hpp).
    const WaveWork work = wave_work<Split>(a, wave, WaveRows);
    if (work.sq == 0) return;                      // empty work item (from device varlen lists)

    // Q rows, staged once the K/V DMA is under way. Rows past the sequence load its row 0 but stage zeros.
    constexpr int RowVecs = HeadDim / 8, QTileLoads = 16 * RowVecs / WaveSize;
    auto q_row = [&](int qt, int i) INLINE {
        return work.row0 + 16 * qt + (i * WaveSize + lane) / RowVecs;
    };
    const bf16_t* const q_rows = qo_rows(a.q, a.q_str, work);
    bf16_t* const o_rows = qo_rows(a.o, a.o_str, work);
    pin_scalar(o_rows);                            // set here, or O's strides would stay live across the loop
    __shared__ LseRows lse_stash[NumWaves];        // LSEs park in LDS: the loop can spare no SGPR
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
    // DMA instruction i of wave w fills K slab (d/64 = i, kv/8 = w) and V slab kv/4 = 2w + i within a
    // tile; lane L writes the slab's 16 B at offset 16L, sourced from the wave's rows 8w .. 8w + 7:
    //   K [kv][d] at [L/8][i, k_chunk(L%8, L/8), 8]        V [kv][d] at [i, L%8/2][L/8, L%2, 8]
    // A row corresponds to a token, whose byte stride never exceeds 8 MB (host::valid_strides).
    const uint32_t k_row_bytes = a.k_str.token * sizeof(bf16_t), v_row_bytes = a.v_str.token * sizeof(bf16_t);
    const auto k_src = lane_view<xdim<P<8>>, xdim<Y<2>, P<8>, Y<8>>>(
        opus::make_tuple(k_row_bytes, 2_I), lane / 8, k_chunk(lane % 8, lane / 8));
    const auto v_src = lane_view<xdim<Y<2>, P<4>>, xdim<P<8>, P<2>, Y<8>>>(
        opus::make_tuple(v_row_bytes, 2_I), lane % 8 / 2, lane / 8, lane % 2);

    // Buffer resources span the wave's rows from tile rsrc_t0 through the sequence's final key; rows
    // beyond it read as 0 rather than the following sequence's (masked P = 0 times a NaN V yields NaN).
    // A tile's byte offset rides in the 32-bit soffset (range-checked): resources advance every
    // rsrc_tiles tiles (2 when loaded ahead; for paged, never, since each block produces the next's).
    const bf16_t* const k_rows = kv_rows(a.k, a.k_str, work);
    const bf16_t* const v_rows = kv_rows(a.v, a.v_str, work);
    const size_t num_rows = work.kv_len - work.tile_lo * KvTile;   // measured from kv_row0
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

    // Paged: rows 8w .. 8w + 7 of a tile fall within a single page (8 divides the page size). The wave's
    // resources begin at the first with soffset 0 and run to the sequence's last key, no more than 256
    // rows long (spanning just 8 rows would slow the DMA by 9%). V's resource equals K's from the prior
    // block shifted by v - k (both pools share strides). k_page_rsrc(i): piece i of the next K resource
    // (rows k_vk + 64, in page k_page) and of the page-after-next's table entry; every piece anchors what
    // it writes.
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
    auto next_page = [&]() INLINE {                // the next block's page (fetched a tile earlier), plus the one after
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

    // Block t issues loads for K(t + k_ahead) and V(t + v_ahead): buffers nobody reads until the next
    // barrier. (A's block 0 redundantly reloads V(0) with identical data.)
    const int k_ahead = group_a ? 1 : 2, v_ahead = group_a ? 0 : 1;

    // Block t's DMA set-up runs entirely in SALU, ahead of the block [salu]: byte offsets and LDS slabs (M0).
    uint32_t k_off = 0, v_off = 0, dma_lds[BlockDmas];
    auto set_dma = [&](int t) INLINE {
        if constexpr (!Paged) {
            k_off = (t - rsrc_t0 + k_ahead) * k_tile_bytes;
            v_off = (t - rsrc_t0 + v_ahead) * v_tile_bytes;
        }
        for (int i = 0; i < DmaInstsPerTile; ++i) {
            dma_lds[i] = (uint32_t)(uintptr_t)(lds_ptr_t)k_slab(t + k_ahead, i);
            dma_lds[DmaInstsPerTile + i] = (uint32_t)(uintptr_t)(lds_ptr_t)v_slab(t + v_ahead, i);
        }
    };

    // Paged: following its DMA, block t builds block t + 1's resources one piece per step (2 for V, then
    // K's), anchoring each there; an SALU op costs 4 issue cycles and a step fits two [salu].
    constexpr int NextRsrcPieces = 2 + KPagePieces;
    auto next_rsrc = [&](int i) INLINE {
        if (i == 0) v_rsrc.x = v_base(), v_rsrc.y = v_base() >> 32, anchor(v_rsrc);
        if (i == 1) v_rsrc.z = k_rsrc.z, anchor(v_rsrc);
        if (i >= 2) k_page_rsrc(i - 2);
    };

    // Block step j's DMA: instruction n is issued at step 4n + 1 [dma], after waiting on that step's
    // fragment (omitting the wait makes LLVM spill). M0 is assigned one step in advance, DMA 0's in
    // prep_block.
    auto dma_step = [&](int j, const bf16x8_t& frag) INLINE {
        const int n = j / 4, i = j - (4 * BlockDmas - 2);   // i: the paged piece, starting the step after
        if (j % 4 == 0 && n > 0 && n < BlockDmas) set_m0(dma_lds[n]);
        if (j % 4 == 1 && n < BlockDmas) {
            if (n < DmaInstsPerTile) buffer_load_dwordx4_lds_m0(k_rsrc, k_src(n), k_off, frag);
            else buffer_load_dwordx4_lds_m0(v_rsrc, v_src(n - DmaInstsPerTile), v_off, frag);
        }
        if (Paged && i >= 0 && i < NextRsrcPieces) next_rsrc(i);
    };

    // Prologue: loads K(0), V(0), and B's portion of K(1) (A's block 0 loads A's portion). Paged: the
    // pages for tiles 0 .. 4 are fetched together (leaving none in flight [lgkm]); each K resource is
    // completed (k_vk advancing one tile), block 0's coming last.
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
    for (int i = 0; i < DmaInstsPerTile; ++i) load_k(0, 0, i);
    for (int i = 0; i < DmaInstsPerTile; ++i) load_v(0, 0, i);
    if (!group_a) {
        if constexpr (Paged) k_rsrc_at(pg[1]);
        for (int i = 0; i < DmaInstsPerTile; ++i) load_k(1, Paged ? 0 : k_tile_bytes, i);
    }
    if constexpr (Paged) {
        v_rsrc_at_k();
        k_vk = work.kv_row0 + 8 * wave + (k_ahead - 1) * KvTile;
        k_rsrc_at(group_a ? pg[1] : pg[2]);
        k_page = group_a ? pg[2] : pg[3], k_next_page = group_a ? pg[3] : pg[4];
    }

    // ==== Q =======================================================================================
    // Q travels through this wave's LDS slot, which arranges d to match K's fragments. Tiles 0 and 1
    // move on into registers; tile 2 remains in the slot and is re-read during QK (registers are full).
    const auto q_store = lane_view<xdim<Y<4>, P<4>>, xdim<P<8>>, xdim<P<2>>, xdim<Y<8>>>(
        QSlotStride, lane / 16, lane % 8, lane / 8 % 2);
    const auto q_read = lane_view<xdim<P<16>>, xdim<Y<4>, P<2>>, xdim<P<2>>, xdim<Y<8>>>(
        QSlotStride, l, g / 2, g % 2);
    bf16_t* const q_slot = lds + KRegion + VRegion + wave * QSlotElems;
    lds_addr_t q_addr = (lds_addr_t)(q_slot + q_read.lane);
    pin_after(q_addr);                             // held in a single register: avoids an add each block
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
    const float threshold = readfirstlane(RescaleThreshold / scale);   // expressed in S units, kept in SGPRs
    const float inv_scale = readfirstlane(1.0f / scale);

    // -frame / scale per row, used as QK's C operand, so S emerges relative to the frame.
    f32x4_t s_seed[QTiles];
    for (int qt = 0; qt < QTiles; ++qt) s_seed[qt] = -Frame0 / scale;

    // P, PV's B operands: p_frag[ks][qt] carries S tiles 2ks and 2ks + 1 in V^T's kv order.
    bf16x8_t p_frag[KvKSteps][QTiles] = {};

    // P's row sums, computed with a sparse MFMA: B stacks two P fragments along k (the first at k < 32);
    // A is all ones with 2:4 sparsity, and its row m retains fragment m % 4 / 2 at positions 0, 1 (even m)
    // or 2, 3 (odd m) of every four k. Thus C0 + C1 and C2 + C3 give the two fragments' row sums.
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
    // K fragment (kt, ks), QK's A operand, rows l and columns 8g .. 8g+7: within the K tile
    // [d/64][kv/8][kv%8][d%64/8 ^ kv%8/4][d%8] it sits at [g%2][kt/2, l/8, kt%2][l%8][ks, k_chunk(g/2, l)][8].
    const auto k_read =
        lane_view<xdim<P<2>>, xdim<Y<2>, P<2>, Y<2>>, xdim<P<8>>, xdim<Y<4>, P<2>>, xdim<Y<8>>>(
            KTileStride, g % 2, l / 8, l % 8, k_chunk(g / 2, l));
    // V^T fragment (ks, dt), PV's A operand: two ds_read_b64_tr_b16 calls (h = 0, 1) on V tile
    // [kv/4][d/16][kv%4][d%16] at [ks, g/2, h, g%2][dt][l/4][l%4, 4]; lane l receives column d = 16dt + l.
    const auto v_read = lane_view<xdim<Y<2>, P<2>, Y<2>, P<2>>, xdim<Y<8>>, xdim<P<4>>, xdim<P<4>, Y<4>>>(
        VTileStride, g / 2, g % 2, l / 4, l % 4);

    // Step s's fragment, fetched from the supplied V and K tile addresses (QK steps with kt = 0 fetch q tile 2 as well).
    auto read_step = [&](lds_addr_t v_addr, lds_addr_t k_addr, BlockStep s) INLINE {
        if (s.pv)
            return join(ds_read_b64_tr_b16(v_addr + v_read.step(s.ks, 0, s.dt)),
                        ds_read_b64_tr_b16(v_addr + v_read.step(s.ks, 1, s.dt)));
        if constexpr (QTiles == 3)
            if (s.kt == 0) q_frag[2][s.ks] = ds_read_b128(q_addr + q_read.step(s.ks));
        return ds_read_b128(k_addr + k_read.step(s.kt / 2, s.kt % 2, s.ks));
    };

    // ==== block start =============================================================================
    // Block t's DMA set-up, read addresses and first two fragments are readied during softmax t - 1, so
    // the block opens directly with its first MFMA. V(t - 1) and K(t) become visible at the barrier
    // preceding softmax t - 1 (A) or block t - 1 (B).
    lds_addr_t v_lane = (lds_addr_t)(lds + KRegion + v_read.lane);   // held whole: v_addr costs one add
    pin_after(v_lane);
    int block_v_buf, block_k_buf, next_v_buf;      // buffers for V(t - 1), K(t), and V(t) in B's reads
    lds_addr_t v_addr, k_addr, v_next;
    bf16x8_t first_frags[Ahead];
    auto prep_block = [&](int t) INLINE {
        set_dma(t);
        set_m0(dma_lds[0]);                        // M0 is written nowhere else
        block_v_buf = v_buf(max(t - 1, 0));        // block 0 (P = 0) reads V(0); others may contain NaNs
        block_k_buf = k_buf(t);
        next_v_buf = v_buf(t);
    };
    auto prep_block_addrs = [&]() INLINE {         // pinned: kept out of the block
        v_addr = v_lane + block_v_buf;
        k_addr = (lds_addr_t)(lds + block_k_buf + k_read.lane);
        v_next = v_lane + next_v_buf;
        pin_after(v_addr);
        pin_after(k_addr);
    };
    auto prep_block_frags = [&]() INLINE {
        for (int i = 0; i < Ahead; ++i) first_frags[i] = read_step(v_addr, k_addr, block_step<QTiles>(i));
    };

    // ==== frames ==================================================================================
    // Max of q tile qt's S within this lane, arranged as a tree (a dependent max chain runs 3x slower).
    // keep_nan: the frame test's max, made NaN by a NaN S. MaskScore, a 17th leaf, turns each node into
    // a 3-operand maximum3 while leaving every frame unchanged.
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
    // Per-row max of every q tile, reduced across the 4 lanes holding that row.
    auto row_maxes = [&](float (&m)[QTiles]) INLINE {
        for (int qt = 0; qt < QTiles; ++qt) m[qt] = lane_max(qt, std::false_type{});
        max_lane_xor32(m);
        max_lane_xor16(m);
    };

    // Frame displacements for q tile qt, scalar fp32 [pk]; shift is measured in S units.
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

    // First tile: a frame sitting more than Frame0 above its row's max underflows P. Should any row have
    // one, all rows' frames shift to Frame0 above their max.
    auto set_frames = [&]() INLINE {
        bool low = false;
        for (int qt = 0; qt < QTiles; ++qt) {
            const float m = lane_max(qt, std::false_type{});
            low |= (m > NegInf) & (m * scale < -Frame0);   // & rather than &&: avoids a per-lane branch
        }
        if (!any_lane(low)) return;
        float m[QTiles];
        row_maxes(m);
        for (int qt = 0; qt < QTiles; ++qt)                 // the max becomes the seed, -Frame0 / scale
            shift_frames(qt, m[qt] > NegInf ? m[qt] - s_seed[qt][0] : 0.0f);
    };

    // Later tiles: when a row's max exceeds its frame by RescaleThreshold, every row in the wave lifts
    // its frame to RescaleHeadroom above its own max. S (in log2 units) updates immediately, as do A's
    // row sums before they end its softmax; O, the seeds and B's row sums update at the next block's
    // start. A NaN or +inf S (from non-finite inputs) follows this path as well: the test's maxes retain
    // NaN.
    bool frames_moved = false;                     // set by the latest softmax
    float m[QTiles];                               // first the test's lane maxes, then the shifts (two arrays spill)
    auto frames_outgrown = [&]() INLINE {
        for (int qt = 0; qt < QTiles; ++qt) m[qt] = lane_max(qt, std::true_type{});   // in S units
        return lanes_above_or_nan(maybe_nan(maximum_n(m)), threshold) != 0;
    };
    // A NaN or +inf S could leak into other rows' sums via the sparse MFMA's zeros (0 x NaN = NaN), so
    // that row's S all score MaskScore, its frame remains, and its row sum becomes NaN, making the row's
    // O NaN (inv_row_sum). bad: the lanes holding such an S, per q tile.
    auto mask_nonfinite = [&](const uint64_t (&bad)[QTiles]) INLINE {
        const int l = fresh_lane_id() & 15;        // not held live on a rare path
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
        uint64_t bad[QTiles];                      // lanes holding a NaN or +inf S (the lane max preserves it)
        for (int qt = 0; qt < QTiles; ++qt) bad[qt] = lanes_above_or_nan(m[qt], MaxFinite);
        if constexpr (!group_a) s_setprio<1>();    // within a rescale window, B's softmax is the long path
        max_lane_xor32(m);
        max_lane_xor16(m);
        for (int qt = 0; qt < QTiles; ++qt) {
            m[qt] = fmaxf(m[qt] * scale + RescaleHeadroom, 0.0f);   // frames only ever move up
            for (int kt = 0; kt < KvMTiles; ++kt)
                for (int i = 0; i < 4; ++i) s_acc[kt][qt][i] -= m[qt];
            if constexpr (group_a) scale_row_sums(qt, __builtin_amdgcn_exp2f(-m[qt]));
        }
        if constexpr (!group_a) s_setprio<0>();
        sched_fence();                             // the compares' SALU runs last [vcc]
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
    // Keys beyond the sequence's end, or past a causal row's last key, score MaskScore.
    auto mask_edge_tile = [&](int kv0) INLINE {
        const int lane = fresh_lane_id(), g = lane >> 4, l = lane & 15;   // not held live on a rare path
        // The kv index of this lane's S value (kt, i), matching s_acc's layout.
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

    // Tiles needing extra softmax work: the first, the masked ones (running to the end), and the last
    // before the buffer resources advance. A single per-tile test covers all three [branch].
    int next_extra = 0;                            // the following such tile
    auto tile_extras = [&](int t) INLINE {
        if (t >= work.num_unmasked) mask_edge_tile((work.tile_lo + t) * KvTile);
        if (t == 0) set_frames();
        move_rsrc(t + 1);
        next_extra = max(t + 1, min(work.num_unmasked, rsrc_t0 + rsrc_tiles - 1));
    };

    // Per-score passes over scalar fp32 values [pk].
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

    // Softmax t: P(t) = exp2(S scale - frame), carried out in three phases. Rare work is out of line [branch].
    auto softmax = [&](auto group_a, int t) INLINE {
        if constexpr (Paged && group_a) next_page();   // B: its next block's first reads are already in flight
        if (unlikely(t >= next_extra)) tile_extras(t);

        // 1. Convert S to log2 units; should any row have outgrown its frame, all frames move.
        frames_moved = frames_outgrown();
        scale_scores();                            // hides the test's latency
        valu_then_salu<2 * LaneScores, 2>();       // all VALU (< 2 per score), then the test's SALU [vcc]
        if (unlikely(frames_moved)) move_frames(group_a);

        // 2. The exps issued back to back [mfma], with the next block's scalar set-up interposed [salu].
        if constexpr (Paged && !group_a) next_page();
        exp_scores();
        prep_block(t + 1);
        interleave<SchedTrans, SchedSalu, LaneScores>();
        sched_fence();

        // 3. P converted to bf16, plus the next block's opening.
        convert_p();
        prep_block_addrs();
        if constexpr (group_a) {                   // B reads its first fragments within its block
            pin_before(p_frag);                    // the reads trail the conversions [lds]
            prep_block_frags();
            sched_fence();
            row_sums();                            // their MFMA time hides the loop's back edge
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

    // A step emits MFMA, read, MFMA, read, MFMA, with every LDS read tucked behind an MFMA.
    auto order_step = [](int reads) INLINE {
        for (int qt = 0; qt < QTiles; ++qt) {
            sched_group<SchedMfma>();
            if (qt < QTiles - 1 && reads > qt) sched_group<SchedDsRead>();
        }
        if (QTiles == 2 && reads > 1) sched_group<SchedDsRead>();   // two MFMAs paired with two reads
    };

    // Waits for this wave's DMA to land and all waves' to be visible. Nothing is scheduled across it.
    auto tile_barrier = []() INLINE {
        s_waitcnt_vmcnt<0>();
        sched_fence();
        s_barrier();
        sched_fence();
    };

    // MFMA block t: fenced steps [fence], each fetching the fragment belonging to the step two ahead.
    auto mfma_block = [&](auto group_a) INLINE {
        if (unlikely(frames_moved)) follow_frames(group_a);   // dual-issues alongside the partner's softmax
        if constexpr (!group_a) {                  // B's block goes first: A's MFMAs fill the gaps
            s_setprio<1>();                        // set before the barrier: B's MFMAs issue on release
            tile_barrier();
            row_sums();                            // for P(t - 1), ahead of its PV
        }

        bf16x8_t frag[BlockSteps];
        for (int i = 0; i < Ahead; ++i) frag[i] = first_frags[i];
#pragma unroll                                     // j has to stay a constant
        for (int j = 0; j < BlockSteps; ++j) {
            const int r = j + Ahead;
            const BlockStep step = block_step<QTiles>(j), ahead = block_step<QTiles>(r % BlockSteps);

            sched_fence();
            dma_step(j, frag[j]);
            sched_fence();

            // Beyond the block's end, B reads the next block's first two fragments while A reads nothing.
            if (r < BlockSteps) frag[r] = read_step(v_addr, k_addr, ahead);
            else if constexpr (!group_a) first_frags[r - BlockSteps] = read_step(v_next, k_addr, ahead);
            mfma_step(step, frag[j]);
            order_step(r < BlockSteps || !group_a ? ahead.reads : 0);
        }
        sched_fence();                             // the MFMAs are issued ahead of A's DMA wait
        if constexpr (group_a) tile_barrier();
        else s_setprio<0>();
    };

    // The block for a tile masked across every row of this wave: just its DMA portion.
    auto dma_only_block = [&](auto group_a, int t) INLINE {
        move_rsrc(t);
        if constexpr (Paged) next_page();
        set_dma(t);                                // supplies k_off and v_off
        if constexpr (!group_a) tile_barrier();
        for (int i = 0; i < DmaInstsPerTile; ++i) load_k(t + k_ahead, k_off, i);
        for (int i = 0; i < DmaInstsPerTile; ++i) load_v(t + v_ahead, v_off, i);
        if constexpr (Paged) for (int i = 0; i < NextRsrcPieces; ++i) next_rsrc(i);
        if constexpr (group_a) tile_barrier();
    };

    // ==== the main loop ===========================================================================
    // A single barrier per tile. Between barriers the SIMD's two waves swap roles on their own:
    //   A:  block 0 | softmax 0, block 1 | softmax 1, block 2 | ...
    //   B:          | block 0, softmax 0 | block 1, softmax 1 | ...
    // Every point at which the groups diverge:
    //                            A                              B
    //   prologue DMA             K(0), V(0)                     K(0), V(0), K(1)
    //   block t's DMA            K(t + 1), V(t)                 K(t + 2), V(t + 1)
    //   tile barrier             after its block                before its block
    //   block t+1's first reads  after softmax t's conversions  in block t's last two steps
    //   row sums of P(t)         end of softmax t               start of block t + 1
    //   issue priority           0                              1 in its block and its frame moves
    if (group_a) s_waitcnt_vmcnt<0>();
    else         s_waitcnt_vmcnt<DmaInstsPerTile>();   // K(1) might still be in flight
    s_barrier();                                       // tile 0 becomes visible to every wave

    // Each group executes its own copy, with group_a a compile-time constant, so no group test [branch].
    auto main_loop = [&](auto group_a) INLINE {
        prep_block(0);
        prep_block_addrs();
        prep_block_frags();                            // B's as well: block 0 has no predecessor
        for (int t = 0; t < work.num_live; ++t) {
            mfma_block(group_a);
            softmax(group_a, t);
        }
        mfma_block(group_a);                           // final PV (its QK is dead code)
        for (int t = work.num_live + 1; t <= work.num_tiles; ++t) dma_only_block(group_a, t);
    };
    if (group_a) main_loop(std::true_type{});
    else         main_loop(std::false_type{});

    // ==== epilogue: O = O^T / row sum =============================================================
    float row_sum[QTiles] = {row_sum01[0] + row_sum01[1], row_sum01[2] + row_sum01[3]};
    if constexpr (QTiles == 3) row_sum[2] = (row_sum2[0] + row_sum2[1]) + (row_sum2[2] + row_sum2[3]);

    if (work.slot >= 0) {                          // a part of a split q block: fp32 partials destined for merge_parts
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
    if constexpr (Split) {                         // one whole q block's LSEs, with a lane per row
        const LseRows lse = lse_stash[wave];
        for (int qt = 0; qt < QTiles; ++qt)
            if (g == 0) store_lse(lse, work, work.row0 + 16 * qt + l, -s_seed[qt][0] * scale, row_sum[qt]);
    }

    // O^T detours through LDS so the global stores write complete rows. It reclaims the K/V region once
    // all waves' final DMA has landed and no wave still reads K or V.
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
    const int id = fresh_lane_id();                    // otherwise the Q load's row addresses stay live and spill
    for (int r = 0; r < WaveRows / RowsPerStore; ++r) {
        const int row = r * RowsPerStore + id / RowVecs;
        const int d   = id % RowVecs * 8;
        const int q   = work.row0 + row;
        if ((unsigned)q < (unsigned)work.sq)
            *reinterpret_cast<bf16x8_t*>(row_at(o_rows, a.o_str, q) + d) =
                *reinterpret_cast<const bf16x8_t*>(o_stage + row * OPitch + d);
    }
}

}  // closes namespace attn

// ---- the C API, declared in attention.h -------------------------------------------------------
using namespace attn;

// The kernels (attention_plan.hpp). A 32-row-wave KV tile costs 0.83 that of a 48-row one (fitted).
static const host::Kernels kernels{{{fwd<2, false>, fwd<2, true>}, {fwd<3, false>, fwd<3, true>}}, 0.83f};
static const host::Kernels paged_kernels{{{fwd<2, false, true>, fwd<2, true, true>},
                                          {fwd<3, false, true>, fwd<3, true, true>}}, 0.83f};

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
