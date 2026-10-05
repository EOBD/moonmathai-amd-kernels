// gfx950 attention forward for MLA prefill (unfused): bf16, Q/K head dim 192 (128 nope + 64 rope) and
// V/O head dim 128; batches, varlen, paged K/V, causal, GQA and LSE as in attention.hip. C API:
// attention.h.
//
// The attention.hip kernel (read that one first) with a wider QK. The differences, tagged below:
//   [k6]    QK takes 6 k-steps: an MFMA block is 40 steps, 16 PV plus 24 QK.
//   [2k]    LDS holds 2 K tile buffers rather than 3. Since both groups' block t reads K(t), their
//           blocks cannot overlap: each block lies between two barriers, and while one group runs its
//           block the other runs its softmax, which is then the long path and issues first. Each group's
//           DMA moves a single tensor (K for A, V for B); both groups' row sums open their blocks.
//   [vgpr]  Q consumes 16 extra VGPRs, so values that attention.hip keeps live are rebuilt or held
//           narrower: DMA offsets in soffset, one s_seed float per q tile, the row sums' A operand,
//           constants and lane ids recomputed, read addresses advanced by SGPR steps.
// Notation and scheduling rules ([mfma], [salu], ...): attention_prims.hpp.

#include "attention.h"
#include "attention_plan.hpp"

namespace attn::mla {

constexpr int Dqk = 192, Dv = 128;                 // head dims of Q/K and of V/O
constexpr int DKSteps = Dqk / 32;                  // k-steps per QK [k6]
constexpr int DMTiles = Dv / 16;                   // PV M-tile count

// LDS layout: a 2-deep K tile ring (tile t in buffer t % 2) [2k], a 3-deep V tile ring (t % 3), and one
// Q slot per wave. A K tile comprises 24 slabs of 512 elements and a V tile 16; each group-A wave fills
// 6 slabs of a K tile, each group-B wave 4 of a V tile.
//   K tile [d/64][kv%8][kv/8][d%64]    V tile [kv/4][d/16][kv%4][d%16]    Q slot [q][d]
// Paged K tile [d/64][kv/8][kv%8][d%64/8 ^ kv%8/2][d%8]: one slab spans 8 consecutive rows, so a wave's
// DMA covers rows 16w .. 16w + 15 of a tile, a single page (contiguous mode keeps the first variant: the
// second would cost its causal grids ~0.5%, a DRAM effect).
// Pitches and the K swizzle render the reads conflict-free: a half-wave's 16-B chunks stay distinct mod 512 B.
constexpr int KSlabs = KvTile * Dqk / SlabElems, VSlabs = KvTile * Dv / SlabElems;
constexpr int GroupWaves = NumWaves / 2;
constexpr int KDmas = KSlabs / GroupWaves, VDmas = VSlabs / GroupWaves;   // each per wave, per tile
constexpr int dma_insts(bool group_a) { return group_a ? KDmas : VDmas; }
constexpr int KPitch = SlabElems + 16, VPitch = SlabElems + 64;
constexpr int KBufs = 2, VBufs = 3;
constexpr int KBufElems = KSlabs * KPitch, VBufElems = VSlabs * VPitch;
constexpr int KRegion = KBufs * KBufElems, VRegion = VBufs * VBufElems;
// Tile t's buffer within its region. The unsigned cast stops the loop's int modulo from becoming a signed divide.
constexpr int k_buf(int t) { return (unsigned)t % KBufs * KBufElems; }
constexpr int v_buf(int t) { return (unsigned)t % VBufs * VBufElems; }
constexpr int QPitch = Dqk + 16, QSlotElems = 16 * QPitch;       // row padding keeps reads conflict-free
constexpr int LdsElems = KRegion + VRegion + NumWaves * QSlotElems;
static_assert(LdsElems * (int)sizeof(bf16_t) <= 163840, "gfx950 allows 160 KB");
constexpr auto KTileStride = opus::make_tuple(opus::number<8 * KPitch>{}, opus::number<KPitch>{}, 64_I, 1_I);
constexpr auto PagedKTileStride =
    opus::make_tuple(opus::number<8 * KPitch>{}, opus::number<KPitch>{}, 64_I, 8_I, 1_I);
constexpr auto VTileStride = opus::make_tuple(opus::number<VPitch>{}, 64_I, 16_I, 1_I);
constexpr auto QSlotStride = opus::make_tuple(opus::number<QPitch>{}, 1_I);
// Position of a paged K row's 8-element chunk c within d%64 (the swizzle).
__device__ constexpr int k_chunk(int c, int kv) { return c ^ (kv % 8 / 2); }
constexpr int OPitch = Dv + 8;                     // O staging (borrows the K/V region)
static_assert(NumWaves * MaxWaveRows * OPitch <= LdsElems, "O staging must fit");

// An MFMA block spans 40 steps [k6]: 16 PV steps (ks, dt) then 24 QK steps (ks, kt). Each step fetches
// one V^T or K fragment (QK steps where kt = 0 also fetch q tile 2) and drives the wave's 3 q tiles.
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

// QTiles: q tiles of 16 rows per wave, 3 or 2 (attention_plan.hpp). Split: the launch includes split q
// blocks whose parts save fp32 partials, or it writes LSEs (otherwise only whole q blocks' O). Paged:
// K/V come as pages.
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
    const int gw = wave % GroupWaves;              // index of the wave inside its group

    // ==== work assignment =========================================================================
    // The WaveRows q rows of one head this wave handles, starting at work.row0, plus its KV tiles
    // (attention_plan.hpp).
    const WaveWork work = wave_work<Split>(a, wave, WaveRows);
    if (work.sq == 0) return;                      // empty work item (from device varlen lists)

    // Q rows, handled as in attention.hip. Load i of a q tile reads its 16-B vector i * WaveSize + lane, 24 per row.
    constexpr int RowVecs = Dqk / 8, QTileLoads = 16 * RowVecs / WaveSize;
    auto q_row = [&](int qt, int i) INLINE {
        return work.row0 + 16 * qt + (i * WaveSize + lane) / RowVecs;
    };
    auto q_vec = [&](int i) INLINE { return (i * WaveSize + lane) % RowVecs; };
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
            q_load[qt][i] = *reinterpret_cast<const bf16x8_t*>(row + 8 * q_vec(i));
        }

    // ==== the K/V DMA =============================================================================
    // Each group's DMA moves a single tensor [2k]: K for A, V for B. DMA instruction i of group wave w
    // fills slab 4i + w of a tile; lane L writes the slab's 16 B at 16L:
    //   K [kv][d] at [L/8, i%2, w][i/2, L%8, 8]        V [kv][d] at [i, w, L%8/2][L/8, L%2, 8]
    // Paged: K slab (d/64 = i/2, kv/8 = 2w + i%2) and V slab 4w + i, taken from the wave's rows
    // 16w .. 16w + 15:
    //   K [kv][d] at [i%2, L/8][i/2, k_chunk(L%8, L/8), 8]        V [kv][d] at [i, L%8/2][L/8, L%2, 8]
    // An instruction's offset from its lane's rides in soffset: a single VGPR, dma_lane [vgpr].
    // A row corresponds to a token, whose byte stride never exceeds 8 MB (host::valid_strides).
    const uint32_t k_row_bytes = a.k_str.token * sizeof(bf16_t), v_row_bytes = a.v_str.token * sizeof(bf16_t);
    const auto k_src = [&]() INLINE {
        const auto stride = opus::make_tuple(k_row_bytes, opus::number<sizeof(bf16_t)>{});
        if constexpr (Paged)
            return lane_view<xdim<Y<2>, P<8>>, xdim<Y<3>, P<8>, Y<8>>>(stride, lane / 8,
                                                                     k_chunk(lane % 8, lane / 8));
        else return lane_view<xdim<P<8>, Y<2>, P<4>>, xdim<Y<3>, P<8>, Y<8>>>(stride, lane / 8, gw, lane % 8);
    }();
    const auto v_src = [&]() INLINE {
        const auto stride = opus::make_tuple(v_row_bytes, opus::number<sizeof(bf16_t)>{});
        if constexpr (Paged)
            return lane_view<xdim<Y<4>, P<4>>, xdim<P<8>, P<2>, Y<8>>>(stride, lane % 8 / 2, lane / 8,
                                                                     lane % 2);
        else return lane_view<xdim<Y<4>, P<4>, P<4>>, xdim<P<8>, P<2>, Y<8>>>(stride, gw, lane % 8 / 2,
                                                                               lane / 8, lane % 2);
    }();
    const uint32_t dma_lane = group_a ? k_src.lane : v_src.lane;

    // Buffer resources spanning tile rsrc_t0 through the sequence's final key, as in attention.hip.
    const bf16_t* const dma_rows = group_a ? kv_rows(a.k, a.k_str, work) : kv_rows(a.v, a.v_str, work);
    const size_t num_rows = work.kv_len - work.tile_lo * KvTile;   // measured from kv_row0
    const uint32_t row_bytes = group_a ? k_row_bytes : v_row_bytes, tile_bytes = KvTile * row_bytes;
    auto dma_rsrc_at = [&](int t) INLINE {
        return kv_rsrc_at(dma_rows, (size_t)t * KvTile, num_rows, group_a ? k_row_bytes : v_row_bytes,
                          (group_a ? Dqk : Dv) * sizeof(bf16_t));
    };
    const int rsrc_tiles = Paged ? INT_MAX / 2 : UINT32_MAX / (KvTile * max(k_row_bytes, v_row_bytes)) - 1;
    int rsrc_t0 = 0;
    std::conditional_t<Paged, rsrc_t, __amdgpu_buffer_rsrc_t> dma_rsrc;   // paged form: its words
    if constexpr (!Paged) dma_rsrc = dma_rsrc_at(0);
    else dma_rsrc.w = 0x00020000u;                 // the configuration word
    auto move_rsrc = [&](int t) INLINE {
        if constexpr (!Paged)
            if (t - rsrc_t0 >= rsrc_tiles) {
                rsrc_t0 = t;
                dma_rsrc = dma_rsrc_at(t);
            }
    };

    // Paged, like K in attention.hip: rows 16w .. 16w + 15 of a tile fall within one page (16 divides
    // the page size); the resource begins at the first (instruction offsets live in soffset).
    // dma_page_rsrc(i): piece i of the next resource (rows dma_vk + 64, in page dma_page) and of the
    // page-after-next's table entry.
    const uint64_t dma_pool = (uint64_t)(group_a ? a.k + offset(a.k_str, 0, 0, work.kv_head)
                                                 : a.v + offset(a.v_str, 0, 0, work.kv_head));
    const uint32_t page_bytes = (group_a ? a.k_str.batch : a.v_str.batch) * 2;
    const uint32_t kv_end = work.kv_row0 + num_rows;   // in virtual tokens
    const uint32_t kv_last = work.kv_len > 0 ? kv_end - 1 : 0;
    uint32_t dma_vk = work.kv_row0 + 16 * gw - KvTile, dma_entry, dma_row;
    int dma_page, dma_next_page, dma_left;         // dma_next_page is in flight
    uint64_t dma_base;
    auto dma_page_rsrc = [&](int i) INLINE {
        if constexpr (Paged) switch (i) {
        case 0: anchor(dma_page), dma_base = (uint64_t)(uint32_t)dma_page * page_bytes;
                anchor(dma_base); break;
        case 1: dma_vk += KvTile, dma_left = (int)(kv_end - dma_vk), anchor(dma_vk), anchor(dma_left); break;
        case 2: dma_row = dma_vk & ~(~0u << a.page_shift), anchor(dma_row); break;   // offset within the page
        case 3: dma_row *= row_bytes, dma_left = max(dma_left, 0), anchor(dma_row), anchor(dma_left); break;
        case 4: dma_left = min(dma_left, 256), anchor(dma_left); break;
        case 5: dma_base += dma_row, anchor(dma_base); break;
        case 6: dma_base += dma_pool, dma_rsrc.x = dma_base, dma_rsrc.y = dma_base >> 32;
                anchor(dma_rsrc); break;
        case 7: dma_rsrc.z = dma_left * row_bytes, dma_entry = dma_vk + 2 * KvTile;
                anchor(dma_rsrc), anchor(dma_entry); break;
        case 8: dma_entry = min(dma_entry, kv_last) >> a.page_shift, anchor(dma_entry); break;
        case 9: dma_entry <<= 2, anchor(dma_entry); break;
        }
    };
    constexpr int PagePieces = 10;
    auto next_page = [&]() INLINE {                // the next block's page (fetched a tile earlier), plus the one after
        wait_page(dma_next_page);
        dma_page = dma_next_page;
        dma_next_page = load_page(a.page_table, dma_entry);
    };

    // DMA instruction i of tile t: its LDS slab together with its byte offset relative to dma_lane's.
    auto dma_slab = [&](auto group_a, int t, int i) INLINE {
        if constexpr (group_a)
            return lds + k_buf(t) + (Paged ? 8 * (i / 2) + 2 * gw + i % 2 : GroupWaves * i + gw) * KPitch;
        else return lds + KRegion + v_buf(t) + (Paged ? 4 * gw + i : GroupWaves * i + gw) * VPitch;
    };
    auto dma_inst_off = [&](auto group_a, int i) INLINE {
        if constexpr (group_a) return k_src.step(i % 2, i / 2);
        else return v_src.step(i);
    };
    auto load_dma = [&](auto group_a, int t, uint32_t off, int i) INLINE {
        const lds_ptr_t slab = (lds_ptr_t)dma_slab(group_a, t, i);
        buffer_load_dwordx4_lds(dma_rsrc, slab, dma_lane, off + dma_inst_off(group_a, i));
    };

    // Block t issues loads for K(t + 1) (group A) or V(t + 1) (group B) [2k]: the buffers of K(t - 1)
    // and V(t - 2), which no later block reads. The wave awaits it at the next block's opening barrier.
    // Its set-up runs entirely in SALU, ahead of the block [salu]: byte offset and LDS slabs (M0).
    uint32_t dma_off = 0, dma_lds[KDmas];
    auto set_dma = [&](auto group_a, int t) INLINE {
        if constexpr (!Paged) dma_off = (t - rsrc_t0 + 1) * tile_bytes;
        for (int i = 0; i < dma_insts(group_a); ++i)
            dma_lds[i] = (uint32_t)(uintptr_t)(lds_ptr_t)dma_slab(group_a, t + 1, i);
    };

    // Block step j's DMA: instruction n is issued at step 4n + 1 [dma], after waiting on that step's
    // fragment (omitting the wait makes LLVM spill). M0 is assigned one step in advance, DMA 0's in
    // prep_block. Paged: once its DMA is done, block t builds block t + 1's resource one piece per step,
    // anchoring each there; an SALU op costs 4 issue cycles and a step fits two [salu].
    auto dma_step = [&](auto group_a, int j, const bf16x8_t& frag) INLINE {
        const int n = j / 4, i = j - (4 * dma_insts(group_a) - 2);   // i: the paged piece
        if (j % 4 == 0 && n > 0 && n < dma_insts(group_a)) set_m0(dma_lds[n]);
        if (j % 4 == 1 && n < dma_insts(group_a))
            buffer_load_dwordx4_lds_m0(dma_rsrc, dma_lane, dma_off + dma_inst_off(group_a, n), frag);
        if (Paged && i >= 0 && i < PagePieces) dma_page_rsrc(i);
    };

    // Prologue: K(0) for A and V(0) for B. Paged: tile 0's resource plus block 0's are completed
    // (dma_vk advancing one tile), their pages fetched together, followed by tile 2's.
    if constexpr (Paged) {
        dma_page = load_page(a.page_table, page_entry(dma_vk + KvTile, kv_last, a.page_shift));
        dma_next_page = load_page(a.page_table, page_entry(dma_vk + 2 * KvTile, kv_last, a.page_shift));
        wait_page(dma_page), wait_page(dma_next_page);
        for (int i = 0; i < PagePieces; ++i) dma_page_rsrc(i);
    }
    if (group_a) for (int i = 0; i < KDmas; ++i) load_dma(std::true_type{}, 0, 0, i);
    else         for (int i = 0; i < VDmas; ++i) load_dma(std::false_type{}, 0, 0, i);
    if constexpr (Paged) {
        dma_page = dma_next_page;
        dma_next_page = load_page(a.page_table, page_entry(dma_vk + 2 * KvTile, kv_last, a.page_shift));
        for (int i = 0; i < PagePieces; ++i) dma_page_rsrc(i);
        next_page();
    }

    // ==== the Q path ==============================================================================
    // Q travels through this wave's LDS slot, reread in QK's B layout. Tiles 0 and 1 move on into
    // registers; tile 2 remains in the slot and is re-read during QK (registers are full).
    auto q_store = [&](int i) INLINE {
        return (i * WaveSize + lane) / RowVecs * QPitch + 8 * q_vec(i);
    };
    const auto q_read = lane_view<xdim<P<16>>, xdim<Y<6>, P<4>, Y<8>>>(QSlotStride, l, g);
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

    // ==== the state ===============================================================================
    f32x4_t o_acc[DMTiles][QTiles] = {};           // O^T(dt, qt)[i] = O^T[d = 16dt + 4g + i][q = 16qt + l]

    // S^T(kt, qt)[i] = S^T[kv][q = 16qt + l], with kv at [kt/2, g/2, kt%2, g%2, i] (C row 4g + i of K(kt)).
    f32x4_t s_acc[KvMTiles][QTiles];
    const float scale = a.scale_log2e;
    const float threshold = readfirstlane(RescaleThreshold / scale);   // expressed in S units, kept in SGPRs
    const float inv_scale = readfirstlane(1.0f / scale);

    // -frame / scale per row, used as QK's C operand, so S emerges relative to the frame. One VGPR per q
    // tile [vgpr]: the C operand's four copies are needed only for QK's first k-step.
    float s_seed[QTiles];                          // assigned in main_loop [vgpr]
    f32x4_t s_seed4[QTiles];                       // s_seed in C-operand form, assigned at each block's start
    const float seed0 = readfirstlane(-Frame0 / scale);

    // P, PV's B operands: p_frag[ks][qt] carries S tiles 2ks and 2ks + 1 in V^T's kv order.
    bf16x8_t p_frag[KvKSteps][QTiles] = {};

    // P's row sums via a sparse MFMA, as in attention.hip; its A operand and selector are rebuilt every
    // block, from a fresh lane id and opaque constants [vgpr].
    bf16x8_t ones;                                 // row l, k 16g .. 16g + 15
    int ones_idx;
    auto set_ones = [&]() INLINE {
        const int id = fresh_lane_id(), g = id >> 4, l = id & 15;
        bf16_t one = 1.0f;
        int odd = (int)0xEEEEEEEEu, even = 0x44444444;
        pin_after(one), pin_after(odd), pin_after(even);
        ones = ((l & 3) < 2) == (g < 2) ? one : (bf16_t)0.0f;
        ones_idx = (l & 1) ? odd : even;
        pin_after(ones);
        pin_after(ones_idx);
    };
    f32x4_t row_sum01{}, row_sum2{};               // q tiles 0 and 1; q tile 2's pair of k-steps
    auto row_sums = [&]() INLINE {
        row_sum01 = v_smfmac_f32_16x16x64_bf16(ones, join(p_frag[0][0], p_frag[0][1]), row_sum01, ones_idx);
        row_sum01 = v_smfmac_f32_16x16x64_bf16(ones, join(p_frag[1][0], p_frag[1][1]), row_sum01, ones_idx);
        if constexpr (QTiles == 3)
            row_sum2 = v_smfmac_f32_16x16x64_bf16(ones, join(p_frag[0][2], p_frag[1][2]), row_sum2, ones_idx);
    };

    // ==== the LDS reads ===========================================================================
    // K fragment (kt, ks), QK's A operand, rows l and columns 8g .. 8g+7: in K tile
    // [d/64][kv%8][kv/8][d%64] it lies at [ks/2][l%8][kt/2, l/8, kt%2][ks%2, g, 8]; half-waves on distinct banks.
    // Paged: within [d/64][kv/8][kv%8][d%64/8 ^ kv%8/2][d%8] it lies at
    //         [ks/2][kt/2, l/8, kt%2][l%8][ks%2, k_chunk(g, l)][8].
    auto k_read_at = [](int g, int l) INLINE {
        if constexpr (Paged)
            return lane_view<xdim<Y<3>>, xdim<Y<2>, P<2>, Y<2>>, xdim<P<8>>, xdim<Y<2>, P<4>>, xdim<Y<8>>>(
                PagedKTileStride, l / 8, l % 8, k_chunk(g, l));
        else return lane_view<xdim<Y<3>>, xdim<P<8>>, xdim<Y<2>, P<2>, Y<2>>, xdim<Y<2>, P<4>, Y<8>>>(
                KTileStride, l % 8, l / 8, g);
    };
    const auto k_read = k_read_at(g, l);
    // V^T fragment (ks, dt), PV's A operand: two ds_read_b64_tr_b16 calls (h = 0, 1) on V tile
    // [kv/4][d/16][kv%4][d%16] at [ks, g/2, h, g%2][dt][l/4][l%4, 4]; lane l receives column d = 16dt + l.
    auto v_read_at = [](int g, int l) INLINE {
        return lane_view<xdim<Y<2>, P<2>, Y<2>, P<2>>, xdim<Y<8>>, xdim<P<4>>, xdim<P<4>, Y<4>>>(
            VTileStride, g / 2, g % 2, l / 4, l % 4);
    };
    const auto v_read = v_read_at(g, l);

    // Step s's fragment, fetched from the supplied V and K tile addresses (QK steps with kt = 0 also fetch q tile 2).
    auto read_step = [&](lds_addr_t v_addr, lds_addr_t k_addr, BlockStep s) INLINE {
        if (s.pv)
            return join(ds_read_b64_tr_b16(v_addr + v_read.step(s.ks, 0, s.dt)),
                        ds_read_b64_tr_b16(v_addr + v_read.step(s.ks, 1, s.dt)));
        if constexpr (QTiles == 3)
            if (s.kt == 0) q_frag[2][s.ks] = ds_read_b128(q_addr + q_read.step(s.ks));
        return ds_read_b128(k_addr + k_read.step(s.ks / 2, s.kt / 2, s.kt % 2, s.ks % 2));
    };

    // ==== starting a block ========================================================================
    // Block t's DMA set-up, read addresses and first two fragments are readied during softmax t - 1, so
    // the block opens directly with its first MFMA. V(t - 1) and K(t) become visible at the barrier
    // preceding block t - 1. Addresses advance block-to-block by an SGPR step, without a pinned lane base
    // [vgpr]; block 0's are assigned in main_loop.
    lds_addr_t v_addr, k_addr, v_next;
    int v_step, k_step, v_next_step;               // shifts from block t - 1's buffers to block t's; V(t)'s is separate
    bf16x8_t first_frags[Ahead];
    auto prep_block = [&](auto group_a, int t) INLINE {
        set_dma(group_a, t);
        set_m0(dma_lds[0]);                        // M0 is written nowhere else
        v_step = t < 2 ? 0 : v_buf(t - 1) - v_buf(t - 2);   // blocks 0 and 1 both read V(0) (P = 0 in block 0)
        k_step = t < 1 ? 0 : k_buf(t) - k_buf(t - 1);
        v_next_step = t < 1 ? 0 : v_buf(t) - v_buf(t - 1);   // B reads block t + 1's start during block t
    };
    auto prep_block_addrs = [&]() INLINE {         // pinned: kept out of the block
        v_addr += v_step;
        k_addr += k_step;
        v_next = v_addr + v_next_step;
        pin_after(v_addr);
        pin_after(k_addr);
    };
    auto prep_block_frags = [&]() INLINE {
        for (int i = 0; i < Ahead; ++i) first_frags[i] = read_step(v_addr, k_addr, block_step<QTiles>(i));
    };

    // ==== the frames ==============================================================================
    // Frames as in attention.hip. Max of q tile qt's S within this lane, arranged as a tree (keep_nan: the test's).
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
        s_seed[qt] -= shift;
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

    // First tile: if some row's frame sits more than Frame0 above its max, all frames shift to Frame0 above.
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
            shift_frames(qt, m[qt] > NegInf ? m[qt] - s_seed[qt] : 0.0f);
    };

    // Later tiles: the lazy rescale. S and A's row sums update immediately; O, the seeds and B's row sums
    // update at the next block's start. A NaN or +inf S follows this path as well.
    bool frames_moved = false;                     // set by the latest softmax
    float m[QTiles];                               // first the test's lane maxes, then the shifts (two arrays spill)
    auto frames_outgrown = [&]() INLINE {
        for (int qt = 0; qt < QTiles; ++qt) m[qt] = lane_max(qt, std::true_type{});   // in S units
        return lanes_above_or_nan(maybe_nan(maximum_n(m)), threshold) != 0;
    };
    // A row containing a NaN or +inf S scores MaskScore, retains its frame and receives a NaN row sum (see
    // attention.hip). bad: the lanes holding such an S, per q tile.
    auto mask_nonfinite = [&](const uint64_t (&bad)[QTiles]) INLINE {
        const int l = fresh_lane_id() & 15;        // not held live on a rare path
        float mask = MaskScore;
        pin_after(mask);                           // not a constant confined to a VGPR [vgpr]
        const float nan = opaque_nan();
        for (int qt = 0; qt < QTiles; ++qt) {
            const bool row_bad = has_row(bad[qt], l);
            for (int kt = 0; kt < KvMTiles; ++kt)
                for (int i = 0; i < 4; ++i) s_acc[kt][qt][i] = row_bad ? mask : s_acc[kt][qt][i];
            m[qt] = row_bad ? 0.0f : m[qt];
            scale_row_sums(qt, row_bad ? nan : 1.0f);
        }
    };
    auto move_frames = [&](auto group_a) INLINE {
        uint64_t bad[QTiles];                      // lanes holding a NaN or +inf S (the lane max preserves it)
        for (int qt = 0; qt < QTiles; ++qt) bad[qt] = lanes_above_or_nan(m[qt], MaxFinite);
        float headroom = RescaleHeadroom;
        pin_after(headroom);                       // not a constant confined to a VGPR [vgpr]
        max_lane_xor32(m);
        max_lane_xor16(m);
        for (int qt = 0; qt < QTiles; ++qt) {
            m[qt] = fmaxf(m[qt] * scale + headroom, 0.0f);   // frames only ever move up
            for (int kt = 0; kt < KvMTiles; ++kt)
                for (int i = 0; i < 4; ++i) s_acc[kt][qt][i] -= m[qt];
            if constexpr (group_a) scale_row_sums(qt, __builtin_amdgcn_exp2f(-m[qt]));
        }
        sched_fence();                             // the compares' SALU runs last [vcc]
        uint64_t any_bad = 0;
        for (int qt = 0; qt < QTiles; ++qt) any_bad |= bad[qt];
        if (unlikely(any_bad != 0)) mask_nonfinite(bad);
    };
    auto follow_frames = [&](auto group_a) INLINE {
        for (int qt = 0; qt < QTiles; ++qt) {
            const float r = __builtin_amdgcn_exp2f(-m[qt]);
            s_seed[qt] -= m[qt] * inv_scale;
            if constexpr (!group_a) scale_row_sums(qt, r);
            scale_o(qt, r);
        }
    };

    // ==== the softmax =============================================================================
    // Keys beyond the sequence's end, or past a causal row's last key, score MaskScore.
    auto mask_edge_tile = [&](int kv0) INLINE {
        const int lane = fresh_lane_id(), g = lane >> 4, l = lane & 15;   // not held live on a rare path
        // The kv index of this lane's S value (kt, i), matching s_acc's layout.
        const auto s_kv = lane_view<xdim<Y<2>, P<2>, Y<2>, P<2>, Y<4>>>(opus::make_tuple(1_I), g / 2, g % 2);
        float mask = MaskScore;
        pin_after(mask);                           // not a constant confined to a VGPR [vgpr]
        for (int qt = 0; qt < QTiles; ++qt) {
            int kv_limit = work.kv_len - 1;
            if (a.causal) kv_limit = min(kv_limit, work.row0 + 16 * qt + l + work.kv_shift);
            kv_limit -= kv0 + s_kv.lane;
            for (int kt = 0; kt < KvMTiles; ++kt)
                for (int i = 0; i < 4; ++i)
                    s_acc[kt][qt][i] = s_kv.step(kt / 2, kt % 2, i) > kv_limit ? mask : s_acc[kt][qt][i];
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
        prep_block(group_a, t + 1);
        interleave<SchedTrans, SchedSalu, LaneScores>();
        sched_fence();

        // 3. P converted to bf16, the next block's opening, and the row sums' A operand [vgpr].
        convert_p();
        prep_block_addrs();
        if constexpr (group_a) {                   // B reads its first fragments within its block
            pin_before(p_frag);                    // the reads trail the conversions [lds]
            prep_block_frags();
        }
        set_ones();
    };

    // ==== the MFMA block ==========================================================================
    auto mfma_step = [&](BlockStep s, bf16x8_t frag) INLINE {
        for (int qt = 0; qt < QTiles; ++qt) {
            if (s.pv) {
                o_acc[s.dt][qt] = v_mfma_f32_16x16x32_bf16(frag, p_frag[s.ks][qt], o_acc[s.dt][qt]);
            } else {
                const f32x4_t c = s.ks == 0 ? s_seed4[qt] : s_acc[s.kt][qt];
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
    // The barrier that closes a block [2k].
    auto block_barrier = []() INLINE {
        sched_fence();
        s_barrier();
        sched_fence();
    };

    // MFMA block t: fenced steps [fence], each fetching the fragment belonging to the step two ahead,
    // bracketed by a tile barrier and a block barrier [2k].
    auto mfma_block = [&](auto group_a) INLINE {
        if (unlikely(frames_moved)) follow_frames(group_a);   // dual-issues alongside the partner's block
        for (int qt = 0; qt < QTiles; ++qt)       // own registers, dead once QK's first k-step is done [vgpr]
            s_seed4[qt] = s_seed[qt], pin_after(s_seed4[qt]);
        tile_barrier();
        row_sums();                                // for P(t - 1), ahead of its PV; a barrier ends softmax [2k]
        s_setprio<0>();                            // the partner's softmax gets issued first [2k]

        bf16x8_t frag[BlockSteps];
        for (int i = 0; i < Ahead; ++i) frag[i] = first_frags[i];
#pragma unroll                                     // j has to stay a constant
        for (int j = 0; j < BlockSteps; ++j) {
            const int r = j + Ahead;
            const BlockStep step = block_step<QTiles>(j), ahead = block_step<QTiles>(r % BlockSteps);

            sched_fence();
            dma_step(group_a, j, frag[j]);
            sched_fence();

            // Beyond the block's end, B reads the next block's first two fragments while A reads nothing.
            if (r < BlockSteps) frag[r] = read_step(v_addr, k_addr, ahead);
            else if constexpr (!group_a) first_frags[r - BlockSteps] = read_step(v_next, k_addr, ahead);
            mfma_step(step, frag[j]);
            order_step(r < BlockSteps || !group_a ? ahead.reads : 0);
        }
        sched_fence();                             // the MFMAs are issued ahead of the barrier
        block_barrier();
        s_setprio<1>();                            // this wave's softmax
    };

    // The block for a tile masked across every row of this wave: just its DMA portion.
    auto dma_only_block = [&](auto group_a, int t) INLINE {
        move_rsrc(t);
        if constexpr (Paged) next_page();
        set_dma(group_a, t);                       // supplies dma_off
        tile_barrier();
        for (int i = 0; i < dma_insts(group_a); ++i) load_dma(group_a, t + 1, dma_off, i);
        if constexpr (Paged) for (int i = 0; i < PagePieces; ++i) dma_page_rsrc(i);
        block_barrier();
    };

    // ==== the main loop ===========================================================================
    // Two barriers per tile [2k]. Between barriers one group runs a block and the other a softmax:
    //   A:  block 0 | softmax 0 | block 1 | softmax 1 | ...
    //   B:          | block 0   | softmax 0 | block 1 | ...
    // Every point at which the groups diverge:
    //                            A                              B
    //   prologue DMA             K(0)                           V(0)
    //   block t's DMA            K(t + 1)                       V(t + 1)
    //   unpaired barrier         after its last block           before its first block
    //   block t+1's first reads  after softmax t's conversions  in block t's last two steps
    s_waitcnt_vmcnt<0>();
    s_barrier();                                       // tile 0 becomes visible to every wave

    // Each group executes its own copy, with group_a a compile-time constant, so no group test [branch].
    // A value only A's copy uses stays live through B's (whose copy comes first): state begins here.
    auto main_loop = [&](auto group_a) INLINE {
        const int id = fresh_lane_id();
        v_addr = (lds_addr_t)(lds + KRegion + v_read_at(id >> 4, id & 15).lane);
        k_addr = (lds_addr_t)(lds + k_read_at(id >> 4, id & 15).lane);
        prep_block(group_a, 0);
        for (int qt = 0; qt < QTiles; ++qt) s_seed[qt] = seed0;
        prep_block_addrs();
        prep_block_frags();
        set_ones();
        if constexpr (!group_a) s_barrier();           // B's blocks trail by one window
        for (int t = 0; t < work.num_live; ++t) {
            mfma_block(group_a);
            softmax(group_a, t);
        }
        mfma_block(group_a);                           // final PV (its QK is dead code)
        for (int t = work.num_live + 1; t <= work.num_tiles; ++t) dma_only_block(group_a, t);
        if constexpr (group_a) s_barrier();            // pairs with B's last block barrier
    };
    if (group_a) main_loop(std::true_type{});
    else         main_loop(std::false_type{});

    // ==== epilogue: dividing O^T by the row sum ===================================================
    float row_sum[QTiles] = {row_sum01[0] + row_sum01[1], row_sum01[2] + row_sum01[3]};
    if constexpr (QTiles == 3) row_sum[2] = (row_sum2[0] + row_sum2[1]) + (row_sum2[2] + row_sum2[3]);
    const int id = fresh_lane_id(), lane_g = id >> 4, lane_l = id & 15;   // no lane value survives the loops [vgpr]

    if (work.slot >= 0) {                          // a part of a split q block: fp32 partials destined for merge_parts
        for (int qt = 0; qt < QTiles; ++qt) {
            const int q = work.row0 + 16 * qt + lane_l;
            if ((unsigned)q >= (unsigned)work.sq) continue;
            const size_t row = (size_t)work.slot * BlockM + wave * WaveRows + 16 * qt + lane_l;
            float* o_row = a.part_o + row * Dv + 4 * lane_g;
            for (int dt = 0; dt < DMTiles; ++dt)
                *reinterpret_cast<f32x4_t*>(o_row + 16 * dt) = o_acc[dt][qt];
            if (lane_g == 0) a.part_ml[row] = float2{-s_seed[qt] * scale, row_sum[qt]};
        }
        return;
    }
    if constexpr (Split) {                         // one whole q block's LSEs, with a lane per row
        const LseRows lse = lse_stash[wave];
        for (int qt = 0; qt < QTiles; ++qt)
            if (lane_g == 0)
                store_lse(lse, work, work.row0 + 16 * qt + lane_l, -s_seed[qt] * scale, row_sum[qt]);
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
            *reinterpret_cast<bf16x4_t*>(o_stage + (16 * qt + lane_l) * OPitch + 16 * dt + 4 * lane_g) = out;
        }
    }
    constexpr int ORowVecs = Dv / 8, RowsPerStore = WaveSize / ORowVecs;
    for (int r = 0; r < WaveRows / RowsPerStore; ++r) {
        const int row = r * RowsPerStore + id / ORowVecs;
        const int d   = id % ORowVecs * 8;
        const int q   = work.row0 + row;
        if ((unsigned)q < (unsigned)work.sq)
            *reinterpret_cast<bf16x8_t*>(row_at(o_rows, a.o_str, q) + d) =
                *reinterpret_cast<const bf16x8_t*>(o_stage + row * OPitch + d);
    }
}

}  // closes namespace attn::mla

// ---- the C API, declared in attention.h -------------------------------------------------------
using namespace attn;

// The kernels (attention_plan.hpp). A 32-row-wave KV tile costs 0.74 that of a 48-row one (fitted).
static const host::Kernels kernels{{{mla::fwd<2, false>, mla::fwd<2, true>},
                                    {mla::fwd<3, false>, mla::fwd<3, true>}}, 0.74f};
static const host::Kernels paged_kernels{{{mla::fwd<2, false, true>, mla::fwd<2, true, true>},
                                          {mla::fwd<3, false, true>, mla::fwd<3, true, true>}}, 0.74f};

extern "C" int cdna4_attn_mla_fwd_bf16_strided(const void* q, const void* k, const void* v, void* o,
                                               float* lse, const int64_t* strides, int B, int H, int HKv,
                                               int Sq, int Skv, int causal, float scale, hipStream_t stream)
{
    return host::api_fwd<mla::Dqk, mla::Dv>(kernels, {q, k, v, o, lse, strides}, B, H, HKv, Sq, Skv, causal,
                                            scale, stream);
}

extern "C" int cdna4_attn_mla_varlen_fwd_bf16_strided(const void* q, const void* k, const void* v, void* o,
                                                      float* lse, const int64_t* strides,
                                                      const int* cu_seqlens_q, const int* cu_seqlens_k,
                                                      int num_seqs, int H, int HKv, int causal, float scale,
                                                      hipStream_t stream)
{
    return host::api_varlen_fwd<mla::Dqk, mla::Dv>(kernels, {q, k, v, o, lse, strides}, cu_seqlens_q,
                                                   cu_seqlens_k, num_seqs, H, HKv, causal, scale, stream);
}

extern "C" int cdna4_attn_mla_paged_fwd_bf16_strided(const void* q, const void* k, const void* v, void* o,
                                                     float* lse, const int64_t* strides,
                                                     const int* cu_seqlens_q, const int* page_table,
                                                     int table_len, const int* page_starts,
                                                     const int* kv_lens, int num_seqs, int page_size, int H,
                                                     int HKv, int causal, float scale, hipStream_t stream)
{
    return host::api_paged_fwd<mla::Dqk, mla::Dv>(paged_kernels, {q, k, v, o, lse, strides},
                                                  cu_seqlens_q, page_table, table_len, page_starts, kv_lens,
                                                  num_seqs, page_size, H, HKv, causal, scale, stream);
}

extern "C" int cdna4_attn_mla_paged_fwd_bf16(const void* q, const void* k, const void* v, void* o,
                                             const int* cu_seqlens_q, const int* page_table, int table_len,
                                             const int* page_starts, const int* kv_lens, int num_seqs,
                                             int page_size, int H, int HKv, int causal, float scale,
                                             hipStream_t stream)
{
    const auto s = host::paged_strides(page_size, H, HKv, mla::Dqk, mla::Dv);
    return cdna4_attn_mla_paged_fwd_bf16_strided(q, k, v, o, nullptr, s.data(), cu_seqlens_q, page_table,
                                                 table_len, page_starts, kv_lens, num_seqs, page_size, H, HKv,
                                                 causal, scale, stream);
}

extern "C" int cdna4_attn_mla_varlen_fwd_bf16_dev_strided(const void* q, const void* k, const void* v,
                                                          void* o, float* lse, const int64_t* strides,
                                                          const int* cu_seqlens_q, const int* cu_seqlens_k,
                                                          const int* host_cu_seqlens_q,
                                                          const int* host_cu_seqlens_k, int num_seqs,
                                                          int max_seqlen_q, int max_seqlen_k, int total_q,
                                                          int total_k, int H, int HKv, int causal,
                                                          float scale, hipStream_t stream)
{
    const host::VarlenCall c{cu_seqlens_q, cu_seqlens_k, host_cu_seqlens_q, host_cu_seqlens_k, num_seqs,
                             max_seqlen_q, max_seqlen_k, total_q, total_k};
    return host::api_varlen_fwd_dev<mla::Dqk, mla::Dv>(kernels, {q, k, v, o, lse, strides}, c, H, HKv, causal,
                                                       scale, stream);
}

extern "C" int cdna4_attn_mla_varlen_plan(void* plan, const int* cu_seqlens_q, const int* cu_seqlens_k,
                                          const int* host_cu_seqlens_q, const int* host_cu_seqlens_k,
                                          int num_seqs, int max_seqlen_q, int max_seqlen_k, int total_q,
                                          int total_k, int H, int HKv, int causal, hipStream_t stream)
{
    const host::VarlenCall c{cu_seqlens_q, cu_seqlens_k, host_cu_seqlens_q, host_cu_seqlens_k, num_seqs,
                             max_seqlen_q, max_seqlen_k, total_q, total_k};
    return host::api_varlen_plan<mla::Dqk, mla::Dv>(plan, kernels, c, H, HKv, causal, stream);
}

extern "C" int cdna4_attn_mla_varlen_run(const void* plan, const void* q, const void* k, const void* v,
                                         void* o, float* lse, const int64_t* strides, float scale,
                                         hipStream_t stream)
{
    return host::api_varlen_run<mla::Dqk, mla::Dv>(plan, kernels, {q, k, v, o, lse, strides}, scale, stream);
}

// Contiguous tensors: BSHD and packed THD.
extern "C" int cdna4_attn_mla_fwd_bf16(const void* q, const void* k, const void* v, void* o, int B, int H,
                                       int HKv, int Sq, int Skv, int causal, float scale, hipStream_t stream)
{
    const auto s = host::contiguous_strides(Sq, Skv, H, HKv, mla::Dqk, mla::Dv);
    return cdna4_attn_mla_fwd_bf16_strided(q, k, v, o, nullptr, s.data(), B, H, HKv, Sq, Skv, causal, scale,
                                           stream);
}

extern "C" int cdna4_attn_mla_varlen_fwd_bf16(const void* q, const void* k, const void* v, void* o,
                                              const int* cu_seqlens_q, const int* cu_seqlens_k, int num_seqs,
                                              int H, int HKv, int causal, float scale, hipStream_t stream)
{
    const auto s = host::contiguous_strides(0, 0, H, HKv, mla::Dqk, mla::Dv);
    return cdna4_attn_mla_varlen_fwd_bf16_strided(q, k, v, o, nullptr, s.data(), cu_seqlens_q, cu_seqlens_k,
                                                  num_seqs, H, HKv, causal, scale, stream);
}

extern "C" int cdna4_attn_mla_varlen_fwd_bf16_dev(const void* q, const void* k, const void* v, void* o,
                                                  const int* cu_seqlens_q, const int* cu_seqlens_k,
                                                  const int* host_cu_seqlens_q, const int* host_cu_seqlens_k,
                                                  int num_seqs, int max_seqlen_q, int max_seqlen_k,
                                                  int total_q, int total_k, int H, int HKv, int causal,
                                                  float scale, hipStream_t stream)
{
    const auto s = host::contiguous_strides(0, 0, H, HKv, mla::Dqk, mla::Dv);
    return cdna4_attn_mla_varlen_fwd_bf16_dev_strided(q, k, v, o, nullptr, s.data(), cu_seqlens_q,
                                                      cu_seqlens_k, host_cu_seqlens_q, host_cu_seqlens_k,
                                                      num_seqs, max_seqlen_q, max_seqlen_k, total_q, total_k,
                                                      H, HKv, causal, scale, stream);
}
