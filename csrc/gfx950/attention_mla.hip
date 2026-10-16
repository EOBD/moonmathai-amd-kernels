// gfx950 bf16 MLA prefill forward: Q/K head dim 192 (128 nope + 64 rope), V/O 128; inputs as attention.hip.
// C API: attention.h. It is the attention.hip kernel with a wider QK; the differences, tagged below:
//   [k6]    6 QK k-steps: a block has 40 steps, 16 PV and 24 QK.
//   [2k]    2 K tile buffers: blocks cannot overlap, so each lies between two barriers, beside the partner's
//           softmax (issued first). A's DMA moves K, B's V; row sums open both groups' blocks.
//   [vgpr]  Q takes 16 more VGPRs: DMA offsets in soffset, one s_seed per q tile, the row-sum operand,
//           constants and lane ids rebuilt, read addresses advanced by SGPR steps.

#include "attention.h"
#include "attention_plan.hpp"

namespace attn::mla {

constexpr int Dqk = 192, Dv = 128;                 // head dims of Q/K and of V/O
constexpr int DKSteps = Dqk / 32;                  // k-steps per QK [k6]
constexpr int DMTiles = Dv / 16;                   // PV M-tiles (16 d each)

// LDS: 2-deep K ring (t % 2) [2k], 3-deep V ring (t % 3) and one Q slot per wave. Slabs of 512 elements: 24
// per K tile (6 per A wave), 16 per V tile (4 per B wave).
//   K tile [d/64][kv%8][kv/8][d%64]    V tile [kv/4][d/16][kv%4][d%16]    Q slot [q][d]
//   Paged K tile [d/64][kv/8][kv%8][d%64/8 ^ kv%8/2][d%8]: a slab is 8 rows, a wave's DMA rows 16w .. 16w + 15.
// Pitches and the K swizzle keep a half-wave's 16-B chunks distinct mod 512 B, so no bank conflicts.
constexpr int KSlabs = KvTile * Dqk / SlabElems, VSlabs = KvTile * Dv / SlabElems;
constexpr int GroupWaves = NumWaves / 2;
constexpr int KDmas = KSlabs / GroupWaves, VDmas = VSlabs / GroupWaves;   // per wave, per tile
constexpr int dma_insts(bool group_a) { return group_a ? KDmas : VDmas; }
constexpr int KPitch = SlabElems + 16, VPitch = SlabElems + 64;
constexpr int KBufs = 2, VBufs = 3;
constexpr int KBufElems = KSlabs * KPitch, VBufElems = VSlabs * VPitch;
constexpr int KRegion = KBufs * KBufElems, VRegion = VBufs * VBufElems;
// Tile t's buffer offset within its region; unsigned, so no signed-divide sequence.
constexpr int k_buf(int t) { return (unsigned)t % KBufs * KBufElems; }
constexpr int v_buf(int t) { return (unsigned)t % VBufs * VBufElems; }
constexpr int QPitch = Dqk + 16, QSlotElems = 16 * QPitch;       // padded: no bank conflicts
constexpr int LdsElems = KRegion + VRegion + NumWaves * QSlotElems;
static_assert(LdsElems * (int)sizeof(bf16_t) <= 163840, "gfx950 allows 160 KB");
constexpr auto KTileStride = opus::make_tuple(opus::number<8 * KPitch>{}, opus::number<KPitch>{}, 64_I, 1_I);
constexpr auto PagedKTileStride =
    opus::make_tuple(opus::number<8 * KPitch>{}, opus::number<KPitch>{}, 64_I, 8_I, 1_I);
constexpr auto VTileStride = opus::make_tuple(opus::number<VPitch>{}, 64_I, 16_I, 1_I);
constexpr auto QSlotStride = opus::make_tuple(opus::number<QPitch>{}, 1_I);
// Slot of a paged K row's 8-element chunk c within d%64 (the swizzle).
__device__ constexpr int k_chunk(int c, int kv) { return c ^ (kv % 8 / 2); }
constexpr int OPitch = Dv + 8;                     // O staging, in the K/V region
static_assert(NumWaves * MaxWaveRows * OPitch <= LdsElems, "O staging must fit");

// Block t [k6]: 16 PV steps (ks, dt), then 24 QK steps (ks, kt), one fragment each, as in attention.hip.
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
    const int gw = wave % GroupWaves;              // the wave's index in its group

    // ==== work ====================================================================================
    // This wave's q rows (from work.row0) and its KV tiles.
    const WaveWork work = wave_work<Split>(a, wave, WaveRows);
    if (work.sq == 0) return;                      // an empty item of a device-built varlen list

    // Q loads as in attention.hip; load i of a q tile reads 16-B vector i*WaveSize + lane (24 per row).
    constexpr int RowVecs = Dqk / 8, QTileLoads = 16 * RowVecs / WaveSize;
    auto q_row = [&](int qt, int i) INLINE {
        return work.row0 + 16 * qt + (i * WaveSize + lane) / RowVecs;
    };
    auto q_vec = [&](int i) INLINE { return (i * WaveSize + lane) % RowVecs; };
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
            q_load[qt][i] = *reinterpret_cast<const bf16x8_t*>(row + 8 * q_vec(i));
        }

    // ==== K/V DMA =================================================================================
    // A's DMA moves K, B's V [2k]. Instruction i of group wave w fills slab 4i + w; lane L writes 16 B at 16L:
    //   K [kv][d] at [L/8, i%2, w][i/2, L%8, 8]        V [kv][d] at [i, w, L%8/2][L/8, L%2, 8]
    // Paged: K slab (d/64 = i/2, kv/8 = 2w + i%2) and V slab 4w + i, from rows 16w .. 16w + 15:
    //   K [kv][d] at [i%2, L/8][i/2, k_chunk(L%8, L/8), 8]        V [kv][d] at [i, L%8/2][L/8, L%2, 8]
    // The lane offset is in one VGPR (dma_lane), the instruction offsets in soffset [vgpr]. Token strides are at
// most 8 MB.
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

    // Buffer resources span tile rsrc_t0 through the sequence's last key, as in attention.hip.
    const bf16_t* const dma_rows = group_a ? kv_rows(a.k, a.k_str, work) : kv_rows(a.v, a.v_str, work);
    const size_t num_rows = work.kv_len - work.tile_lo * KvTile;   // counted from kv_row0
    const uint32_t row_bytes = group_a ? k_row_bytes : v_row_bytes, tile_bytes = KvTile * row_bytes;
    auto dma_rsrc_at = [&](int t) INLINE {
        return kv_rsrc_at(dma_rows, (size_t)t * KvTile, num_rows, group_a ? k_row_bytes : v_row_bytes,
                          (group_a ? Dqk : Dv) * sizeof(bf16_t));
    };
    const int rsrc_tiles = Paged ? INT_MAX / 2 : UINT32_MAX / (KvTile * max(k_row_bytes, v_row_bytes)) - 1;
    int rsrc_t0 = 0;
    std::conditional_t<Paged, rsrc_t, __amdgpu_buffer_rsrc_t> dma_rsrc;   // paged: as words
    if constexpr (!Paged) dma_rsrc = dma_rsrc_at(0);
    else dma_rsrc.w = 0x00020000u;                 // the configuration word
    auto move_rsrc = [&](int t) INLINE {
        if constexpr (!Paged)
            if (t - rsrc_t0 >= rsrc_tiles) {
                rsrc_t0 = t;
                dma_rsrc = dma_rsrc_at(t);
            }
    };

    // Paged, as K in attention.hip: rows 16w .. 16w + 15 of a tile lie in one page (16 divides the page size).
    // dma_page_rsrc(i): piece i of the next resource (rows dma_vk + 64, page dma_page) and of the
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
    // Take the next block's page (loaded a tile earlier), then prefetch the one after it.
    auto next_page = [&]() INLINE {
        wait_page(dma_next_page);
        dma_page = dma_next_page;
        dma_next_page = load_page(a.page_table, dma_entry);
    };

    // DMA instruction i of tile t: its LDS slab and its byte offset from dma_lane.
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

    // Block t loads K(t + 1) (A) or V(t + 1) (B) [2k] into the buffers of K(t - 1) and V(t - 2); it lands by
    // the next block's barrier. DMA registers set in SALU before the block [salu]: soffset and slab addresses
    // (M0).
    uint32_t dma_off = 0, dma_lds[KDmas];
    auto set_dma = [&](auto group_a, int t) INLINE {
        if constexpr (!Paged) dma_off = (t - rsrc_t0 + 1) * tile_bytes;
        for (int i = 0; i < dma_insts(group_a); ++i)
            dma_lds[i] = (uint32_t)(uintptr_t)(lds_ptr_t)dma_slab(group_a, t + 1, i);
    };

    // Block step j's DMA [dma]: instruction n's M0 at step 4n (DMA 0's in prep_block), its load at 4n + 1,
    // ordered after the step's fragment. Paged: then block t + 1's resource, one piece per step [salu].
    auto dma_step = [&](auto group_a, int j, const bf16x8_t& frag) INLINE {
        const int n = j / 4, i = j - (4 * dma_insts(group_a) - 2);   // i: the paged piece
        if (j % 4 == 0 && n > 0 && n < dma_insts(group_a)) set_m0(dma_lds[n]);
        if (j % 4 == 1 && n < dma_insts(group_a))
            buffer_load_dwordx4_lds_m0(dma_rsrc, dma_lane, dma_off + dma_inst_off(group_a, n), frag);
        if (Paged && i >= 0 && i < PagePieces) dma_page_rsrc(i);
    };

    // Prologue: K(0) for A, V(0) for B. Paged: tile 0's and block 0's resources, with their pages loaded together.
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

    // ==== Q =======================================================================================
    // Q passes through the wave's LDS slot into QK's B layout. Tiles 0 and 1 stay in registers; tile 2 is
    // re-read from the slot during QK.
    auto q_store = [&](int i) INLINE {
        return (i * WaveSize + lane) / RowVecs * QPitch + 8 * q_vec(i);
    };
    const auto q_read = lane_view<xdim<P<16>>, xdim<Y<6>, P<4>, Y<8>>>(QSlotStride, l, g);
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

    // QK's C operand, -frame / scale per row: one VGPR per q tile [vgpr], widened per block for QK's first k-step.
    float s_seed[QTiles];                          // set in main_loop [vgpr]
    f32x4_t s_seed4[QTiles];                       // s_seed as a C operand, set at each block's start
    const float seed0 = readfirstlane(-Frame0 / scale);

    // PV's B: p_frag[ks][qt] holds S tiles 2ks and 2ks + 1 in V^T's kv order.
    bf16x8_t p_frag[KvKSteps][QTiles] = {};

    // Row sums of P by sparse MFMA, as in attention.hip; A and its selector rebuilt per block [vgpr].
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

    // ==== LDS reads ===============================================================================
    // K fragment (kt, ks), QK's A: row l, columns 8g .. 8g + 7, in the K tile at
    // [ks/2][l%8][kt/2, l/8, kt%2][ks%2, g, 8]; paged: [ks/2][kt/2, l/8, kt%2][l%8][ks%2, k_chunk(g, l)][8].
    auto k_read_at = [](int g, int l) INLINE {
        if constexpr (Paged)
            return lane_view<xdim<Y<3>>, xdim<Y<2>, P<2>, Y<2>>, xdim<P<8>>, xdim<Y<2>, P<4>>, xdim<Y<8>>>(
                PagedKTileStride, l / 8, l % 8, k_chunk(g, l));
        else return lane_view<xdim<Y<3>>, xdim<P<8>>, xdim<Y<2>, P<2>, Y<2>>, xdim<Y<2>, P<4>, Y<8>>>(
                KTileStride, l % 8, l / 8, g);
    };
    const auto k_read = k_read_at(g, l);
    // V^T fragment (ks, dt), PV's A: as in attention.hip.
    auto v_read_at = [](int g, int l) INLINE {
        return lane_view<xdim<Y<2>, P<2>, Y<2>, P<2>>, xdim<Y<8>>, xdim<P<4>>, xdim<P<4>, Y<4>>>(
            VTileStride, g / 2, g % 2, l / 4, l % 4);
    };
    const auto v_read = v_read_at(g, l);

    // Step s's fragment (QK kt = 0 also reads q tile 2).
    auto read_step = [&](lds_addr_t v_addr, lds_addr_t k_addr, BlockStep s) INLINE {
        if (s.pv)
            return join(ds_read_b64_tr_b16(v_addr + v_read.step(s.ks, 0, s.dt)),
                        ds_read_b64_tr_b16(v_addr + v_read.step(s.ks, 1, s.dt)));
        if constexpr (QTiles == 3)
            if (s.kt == 0) q_frag[2][s.ks] = ds_read_b128(q_addr + q_read.step(s.ks));
        return ds_read_b128(k_addr + k_read.step(s.ks / 2, s.kt / 2, s.kt % 2, s.ks % 2));
    };

    // ==== next block's set-up =====================================================================
    // Prepared in softmax t - 1, as in attention.hip. Addresses advance by SGPR steps [vgpr]; block 0's are set
    // in main_loop.
    lds_addr_t v_addr, k_addr, v_next;
    int v_step, k_step, v_next_step;               // buffers from block t - 1 to block t, and on to V(t)
    bf16x8_t first_frags[ReadLead];
    auto prep_block = [&](auto group_a, int t) INLINE {
        set_dma(group_a, t);
        set_m0(dma_lds[0]);                        // DMA 0's M0; the block sets the others
        v_step = t < 2 ? 0 : v_buf(t - 1) - v_buf(t - 2);   // blocks 0 and 1 read V(0) (P = 0 in block 0)
        k_step = t < 1 ? 0 : k_buf(t) - k_buf(t - 1);
        v_next_step = t < 1 ? 0 : v_buf(t) - v_buf(t - 1);   // B reads block t + 1's start in block t
    };
    auto prep_block_addrs = [&]() INLINE {         // pinned: computed here, not in the block
        v_addr += v_step;
        k_addr += k_step;
        v_next = v_addr + v_next_step;
        pin_after(v_addr);
        pin_after(k_addr);
    };
    auto prep_block_frags = [&]() INLINE {
        for (int i = 0; i < ReadLead; ++i) first_frags[i] = read_step(v_addr, k_addr, block_step<QTiles>(i));
    };

    // ==== frames ==================================================================================
    // As in attention.hip. q tile qt's largest S in this lane (keep_nan: the frame test's).
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

    // First tile: if any row's frame is more than Frame0 above its max, every frame moves to Frame0 above its own.
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
            shift_frames(qt, m[qt] > NegInf ? m[qt] - s_seed[qt] : 0.0f);
    };

    // Later tiles: the lazy rescale. S and A's row sums update now; O, seeds and B's row sums update at the next
    // block's start. A NaN or +inf S takes this path too.
    bool frames_moved = false;                     // set by the latest softmax
    float m[QTiles];                               // the test's lane maxima, then the shifts
    auto frames_outgrown = [&]() INLINE {
        for (int qt = 0; qt < QTiles; ++qt) m[qt] = lane_max(qt, std::true_type{});   // in S units
        return lanes_above_or_nan(maybe_nan(maximum_n(m)), threshold) != 0;
    };
    // A row with a NaN or +inf S: S = MaskScore, frame kept, row sum NaN (as in attention.hip).
    auto mask_nonfinite = [&](const uint64_t (&bad)[QTiles]) INLINE {
        const int l = fresh_lane_id() & 15;        // not live outside this rare path
        float mask = MaskScore;
        pin_after(mask);                           // no VGPR held for the constant [vgpr]
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
        uint64_t bad[QTiles];                      // lanes holding a NaN or +inf S
        for (int qt = 0; qt < QTiles; ++qt) bad[qt] = lanes_above_or_nan(m[qt], MaxFinite);
        float headroom = RescaleHeadroom;
        pin_after(headroom);                       // no VGPR held for the constant [vgpr]
        max_lane_xor32(m);
        max_lane_xor16(m);
        for (int qt = 0; qt < QTiles; ++qt) {
            m[qt] = fmaxf(m[qt] * scale + headroom, 0.0f);   // frames only ever move up
            for (int kt = 0; kt < KvMTiles; ++kt)
                for (int i = 0; i < 4; ++i) s_acc[kt][qt][i] -= m[qt];
            if constexpr (group_a) scale_row_sums(qt, __builtin_amdgcn_exp2f(-m[qt]));
        }
        sched_fence();                             // the compares' SALU last [vcc]
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

    // ==== softmax =================================================================================
    // Keys past the sequence or past a causal row's last key: S = MaskScore.
    auto mask_scores = [&](int kv0) INLINE {
        const int lane = fresh_lane_id(), g = lane >> 4, l = lane & 15;   // not live outside this rare path
        // kv of this lane's S value (kt, i).
        const auto s_kv = lane_view<xdim<Y<2>, P<2>, Y<2>, P<2>, Y<4>>>(opus::make_tuple(1_I), g / 2, g % 2);
        float mask = MaskScore;
        pin_after(mask);                           // no VGPR held for the constant [vgpr]
        for (int qt = 0; qt < QTiles; ++qt) {
            int kv_limit = work.kv_len - 1;
            if (a.causal) kv_limit = min(kv_limit, work.row0 + 16 * qt + l + work.kv_shift);
            kv_limit -= kv0 + s_kv.lane;
            for (int kt = 0; kt < KvMTiles; ++kt)
                for (int i = 0; i < 4; ++i)
                    s_acc[kt][qt][i] = s_kv.step(kt / 2, kt % 2, i) > kv_limit ? mask : s_acc[kt][qt][i];
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
        prep_block(group_a, t + 1);
        for (int i = 0; i < LaneScores; ++i) {     // exp, SALU, exp, SALU, ...
            __builtin_amdgcn_sched_group_barrier(SchedTrans, 1, 0);
            __builtin_amdgcn_sched_group_barrier(SchedSalu, 1, 0);
        }
        sched_fence();

        // 3. P to bf16; the next block's addresses (A: also its first fragments); the row sums' A [vgpr].
        convert_p();
        prep_block_addrs();
        if constexpr (group_a) {                   // B reads its first fragments in its block
            pin_before(p_frag);                    // reads after the conversions [lds]
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
    // The barrier that closes a block [2k].
    auto block_barrier = []() INLINE {
        sched_fence();
        s_barrier();
        sched_fence();
    };

    // Block t between a tile barrier and a block barrier [2k]; steps read ReadLead steps ahead and are fenced [fence].
    auto mfma_block = [&](auto group_a) INLINE {
        if (unlikely(frames_moved)) follow_frames(group_a);   // beside the partner's block
        for (int qt = 0; qt < QTiles; ++qt)       // dead after QK's first k-step [vgpr]
            s_seed4[qt] = s_seed[qt], pin_after(s_seed4[qt]);
        tile_barrier();
        row_sums();                                // of P(t - 1), ahead of its PV [2k]
        s_setprio<0>();                            // the partner's softmax issues first [2k]

        bf16x8_t frag[BlockSteps];
        for (int i = 0; i < ReadLead; ++i) frag[i] = first_frags[i];
#pragma unroll                                     // j has to stay a constant
        for (int j = 0; j < BlockSteps; ++j) {
            const int r = j + ReadLead;
            const BlockStep step = block_step<QTiles>(j), ahead = block_step<QTiles>(r % BlockSteps);

            sched_fence();
            dma_step(group_a, j, frag[j]);
            sched_fence();

            // Past the block end, B reads the next block's first fragments; A reads nothing.
            if (r < BlockSteps) frag[r] = read_step(v_addr, k_addr, ahead);
            else if constexpr (!group_a) first_frags[r - BlockSteps] = read_step(v_next, k_addr, ahead);
            mfma_step(step, frag[j]);
            order_step(r < BlockSteps || !group_a ? ahead.reads : 0);
        }
        sched_fence();                             // MFMAs before the barrier
        block_barrier();
        s_setprio<1>();                            // this wave's softmax
    };

    // Block of a tile masked for every row of this wave: its DMA only.
    auto dma_only_block = [&](auto group_a, int t) INLINE {
        move_rsrc(t);
        if constexpr (Paged) next_page();
        set_dma(group_a, t);                       // dma_off
        tile_barrier();
        for (int i = 0; i < dma_insts(group_a); ++i) load_dma(group_a, t + 1, dma_off, i);
        if constexpr (Paged) for (int i = 0; i < PagePieces; ++i) dma_page_rsrc(i);
        block_barrier();
    };

    // ==== the main loop ===========================================================================
    // Two barriers per tile [2k]; between them one group runs a block, the other a softmax:
//   A:  block 0 | softmax 0 | block 1 | softmax 1 | ...
//   B:          | block 0 | softmax 0 | block 1 | ...
//                            A                              B
    //   prologue DMA             K(0)                           V(0)
    //   block t's DMA            K(t + 1)                       V(t + 1)
    //   unpaired barrier         after its last block           before its first block
    //   block t+1's first reads  after softmax t's conversions  in block t's last two steps
    s_waitcnt_vmcnt<0>();
    s_barrier();                                       // make tile 0 visible to every wave

    // One copy per group; group_a is a compile-time constant [branch]. State is set inside: a value set
    // before stays live through both copies.
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
        mfma_block(group_a);                           // the final PV (its QK goes unused)
        for (int t = work.num_live + 1; t <= work.num_tiles; ++t) dma_only_block(group_a, t);
        if constexpr (group_a) s_barrier();            // pairs with B's last block barrier
    };
    if (group_a) main_loop(std::true_type{});
    else         main_loop(std::false_type{});

    // ==== epilogue: O = O^T / row sum =============================================================
    float row_sum[QTiles] = {row_sum01[0] + row_sum01[1], row_sum01[2] + row_sum01[3]};
    if constexpr (QTiles == 3) row_sum[2] = (row_sum2[0] + row_sum2[1]) + (row_sum2[2] + row_sum2[3]);
    const int id = fresh_lane_id(), lane_g = id >> 4, lane_l = id & 15;   // recomputed [vgpr]

    if (work.slot >= 0) {                          // a part of a split q block: fp32 partials for merge_parts
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
    if constexpr (Split) {                         // LSEs of a whole q block, one lane per row
        const LseRows lse = lse_stash[wave];
        for (int qt = 0; qt < QTiles; ++qt)
            if (lane_g == 0)
                store_lse(lse, work, work.row0 + 16 * qt + lane_l, -s_seed[qt] * scale, row_sum[qt]);
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

}  // namespace attn::mla

// ---- the C API, declared in attention.h -------------------------------------------------------
using namespace attn;

// The kernels (attention_plan.hpp); a 32-row wave's KV tile costs 0.60 of a 48-row one.
static const host::Kernels kernels{{{mla::fwd<2, false>, mla::fwd<2, true>},
                                    {mla::fwd<3, false>, mla::fwd<3, true>}}, 0.60f, {1.0f, 1.0f}};
static const host::Kernels paged_kernels{{{mla::fwd<2, false, true>, mla::fwd<2, true, true>},
                                          {mla::fwd<3, false, true>, mla::fwd<3, true, true>}}, 0.60f,
                                         {1.0f, 1.0f}};

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
