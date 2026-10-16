// Work division and launch for the attention kernels (C API: attention.h).
// Device: the work list (one WorkItem per CTA), a wave's rows and KV tiles (wave_work), K/V resources and pages,
// merge_parts (joins a split q block's partials) and list_varlen (the list from device cu_seqlens).
// Host (namespace host): plan -> list -> launch; lists cached per stream, graph-capture safe; C API checks.
#pragma once

#include "attention_prims.hpp"
#include <algorithm>
#include <array>
#include <climits>
#include <cstddef>
#include <cstring>
#include <deque>
#include <functional>
#include <map>
#include <mutex>
#include <queue>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace attn {

__host__ __device__ constexpr int ceil_div(int a, int b) { return (a + b - 1) / b; }

// ---- tensors ------------------------------------------------------------------------------------
// Every tensor is [batch][token][head][d] with its own strides in elements (d contiguous; BSHD, BHSD, THD).
// A sequence holds Q/O tokens q0 .. q0 + sq - 1 and K/V tokens kv0 .. kv0 + skv - 1 of batch b; batch calls have
// one per batch (q0 = kv0 = 0), varlen calls pack them all in batch 0. Causal is bottom-right aligned.
struct Strides { int64_t batch, token, head; };

// Element offset of token `token`, head `head`, batch `b`.
__device__ __forceinline__ uint64_t offset(const Strides& s, uint32_t b, uint32_t token, uint32_t head)
{
    return b * (uint64_t)s.batch + token * (uint64_t)(uint32_t)s.token + head * (uint64_t)s.head;
}

// ---- the CTA work list --------------------------------------------------------------------------
// A CTA's work: a q block of one sequence and head pack (see rows), whole or one part of its KV tiles.
// A merge item is part 0 of a split q block with num_parts; sq = 0 marks an empty item.
struct WorkItem {
    int q0, sq;                     // the sequence's first Q/O token and q row count
    int kv0, skv;                   // the sequence's first K/V token and key count
    int chunk0;                     // first chunk of the q block (see rows)
    int tile_lo, tile_hi;           // its KV tiles
    int head0;                      // first q head of the head pack
    int b;                          // the sequence's batch
    int slot;                       // -1: a whole q block (stores O); else the slot of its fp32 partials
    int num_parts;                  // merge items: the part count (slots slot, slot + 1, ...)
};
static_assert(offsetof(WorkItem, slot) == 9 * sizeof(int), "fwd<false> reads the first 9 ints");

// A split q block's parts take consecutive slots of BlockM partial rows.
struct Args {
    const bf16_t* __restrict__ q;   // [batch][token][H][Dqk]  (see tensors)
    const bf16_t* __restrict__ k;   // [batch][token][HKv][Dqk]
    const bf16_t* __restrict__ v;   // [batch][token][HKv][Dv]
    bf16_t* __restrict__ o;         // [batch][token][H][Dv]
    Strides q_str, k_str, v_str, o_str;
    const WorkItem* __restrict__ work;   // one item per CTA
    float* part_o;                  // [slot][BlockM][Dv]: a part's unnormalised O
    float2* part_ml;                // [slot][BlockM]: per-row frame m and row sum l
    int H, HKv, grp, pack, causal;  // grp = H / HKv; pack: q heads per head pack (1, or grp)
    float scale_log2e;              // softmax_scale * log2(e)
    float* __restrict__ lse;        // nullptr, or [batch][token][H] at lse_str, each q row's lse
    Strides lse_str;
    const int* page_table;          // paged calls (see paged K/V): table_len entries, pages of 2^page_shift
    int table_len, page_shift;
};

// ---- rows ---------------------------------------------------------------------------------------
// A chunk is a wave's `rows` (48 or 32) consecutive q rows of one head; 8 chunks make a q block (a CTA's work).
// A head pack is the heads sharing q blocks (one, or all of a KV head under causal GQA); chunk c is q chunk
// c / pack of head c % pack. Chunks and q blocks align to the sequence's end: q block 0 gets the empty ones.
__host__ __device__ inline int head_chunks(int sq, int rows) { return ceil_div(sq, rows); }
__host__ __device__ inline int chunk_q0(int c, int sq, int pack, int rows)
{
    return c / pack * rows - (head_chunks(sq, rows) * rows - sq);
}
__host__ __device__ inline int num_q_blocks(int sq, int pack, int rows)
{
    return ceil_div(head_chunks(sq, rows) * pack, NumWaves);
}
__host__ __device__ inline int first_chunk(int b, int sq, int pack, int rows)
{
    return head_chunks(sq, rows) * pack - (num_q_blocks(sq, pack, rows) - b) * NumWaves;
}

// The KV tiles q block b walks: under causal (bottom-right) it stops at the last key of its last row.
__host__ __device__ inline int block_tiles(int sq, int skv, int b, int pack, int rows, int causal)
{
    const int last_chunk = first_chunk(b, sq, pack, rows) + NumWaves - 1;
    int kv_end = skv;
    if (causal) kv_end = min(kv_end, chunk_q0(last_chunk, sq, pack, rows) + rows + skv - sq);
    return ceil_div(max(kv_end, 0), KvTile);
}

// Part j of n of t tiles: tiles j t / n .. (j + 1) t / n - 1.
__host__ __device__ inline int part_lo(int j, int n, int t) { return j * t / n; }
__host__ __device__ inline int part_len(int j, int n, int t) { return (j + 1) * t / n - j * t / n; }

// Chunk w of a work item: its q head and first row (row0 -rows marks an empty chunk).
struct ChunkRows { int head, row0; };
__device__ __forceinline__ ChunkRows chunk_rows(const Args& a, const WorkItem& it, int w, int rows)
{
    const int c = it.chunk0 + w;
    if (c < 0) return {it.head0, -rows};
    return {it.head0 + c % a.pack, chunk_q0(c, it.sq, a.pack, rows)};
}

// ---- the work of one wave -----------------------------------------------------------------------
struct WaveWork {
    int head, kv_head;
    int b;                    // the sequence's batch
    int q0, sq;               // the sequence's first Q/O token and q row count
    int kv_len, kv_shift;     // the sequence's keys; causal: row q's last key is q + kv_shift
    size_t kv_row0;           // first tile's K/V token
    int row0;                 // q of its first row (-rows: all padding)
    int tile_lo, num_tiles;   // its KV tiles tile_lo .. tile_lo + num_tiles - 1
    int num_unmasked;         // tiles from here on need a mask
    int num_live;             // tiles from here on mask every row: DMA only
    int slot;                 // -1, or the slot of its partials
};
template <bool Split>
__device__ __forceinline__ WaveWork wave_work(const Args& a, int wave, int rows)
{
    // 1. The CTA's work item; a split launch's slot and partials loaded here, not in the loops [lgkm].
    const WorkItem it = a.work[blockIdx.x];
    if constexpr (Split) pin_scalar(it.slot), pin_scalar(a.part_o), pin_scalar(a.part_ml);

    // 2. This wave's chunk; head and row0 in SGPRs.
    const ChunkRows c = chunk_rows(a, it, wave, rows);
    const int head = __builtin_amdgcn_readfirstlane(c.head);
    const int row0 = __builtin_amdgcn_readfirstlane(c.row0);

    // 3. Masks: the last key and, under causal, the diagonals of the first row (unmasked) and last row (live).
    const int num_tiles = it.tile_hi - it.tile_lo, kv_shift = it.skv - it.sq;
    int num_unmasked = it.skv / KvTile;
    if (a.causal) num_unmasked = min(num_unmasked, (row0 + kv_shift + 1) / KvTile);
    int num_live = row0 + rows > 0 ? num_tiles : 0;
    if (a.causal) {
        const int last_kv = row0 + rows - 1 + kv_shift;
        num_live = last_kv < 0 ? 0 : max(0, min(num_tiles, last_kv / KvTile + 1 - it.tile_lo));
    }
    const size_t kv_row0 = (size_t)it.kv0 + (size_t)it.tile_lo * KvTile;
    return {head, it.head0 / a.grp, it.b, it.q0, it.sq, it.skv, kv_shift, kv_row0, row0, it.tile_lo,
            num_tiles, max(0, num_unmasked - it.tile_lo), num_live, Split ? it.slot : -1};
}

// The wave's q = 0 row of Q or O, and its kv_row0 row of K or V; row_at advances r token strides.
template <class T>
__device__ __forceinline__ T* qo_rows(T* x, const Strides& s, const WaveWork& w)
{
    return x + offset(s, w.b, w.q0, w.head);
}
__device__ __forceinline__ const bf16_t* kv_rows(const bf16_t* x, const Strides& s, const WaveWork& w)
{
    return x + offset(s, w.b, w.kv_row0, w.kv_head);
}
template <class T>
__device__ __forceinline__ T* row_at(T* rows, const Strides& s, uint32_t r)
{
    return rows + (size_t)r * (uint32_t)s.token;
}

// A K/V buffer resource from row `row` of x's `rows`: its range ends after the last row's d_bytes (at most
// 4 GB), so later rows read 0.
__device__ __forceinline__ __amdgpu_buffer_rsrc_t kv_rsrc_at(const bf16_t* x, size_t row, size_t rows,
                                                            uint32_t row_bytes, uint32_t d_bytes)
{
    const size_t left = row < rows ? rows - row : 0;
    const size_t bytes = left > 0 ? (left - 1) * row_bytes + d_bytes : 0;
    return opus::make_buffer_rsrc((const char*)x + row * row_bytes, (uint32_t)min(bytes, (size_t)UINT32_MAX));
}
// The same resource as words.
__device__ __forceinline__ rsrc_t kv_rsrc_words(const bf16_t* x, size_t row, size_t rows, uint32_t row_bytes,
                                                uint32_t d_bytes)
{
    const size_t left = row < rows ? rows - row : 0;
    const size_t bytes = left > 0 ? (left - 1) * row_bytes + d_bytes : 0;
    const uintptr_t base = (uintptr_t)((const char*)x + row * row_bytes);
    const uint32_t range = (uint32_t)min(bytes, (size_t)UINT32_MAX);
    return rsrc_t{(uint32_t)base, (uint32_t)(base >> 32), range, 0x00020000u};
}

// ---- paged K/V ----------------------------------------------------------------------------------
// K and V are page pools (page stride: k_str.batch, v_str.batch). Key r is virtual token vk = kv0 + r, at row
// vk % page size of page page_table[vk >> page_shift].

// The page at byte `entry` of the table, and the wait for it; asm [lgkm].
__device__ __forceinline__ int load_page(const int* table, uint32_t entry)
{
    int page;
    asm volatile("s_load_dword %0, %1, %2" : "=s"(page) : "s"(table), "s"(entry));
    return page;
}
__device__ __forceinline__ void wait_page(int& page)
{
    asm volatile("s_waitcnt lgkmcnt(0)" : "+s"(page));
}
// The table byte of virtual token vk's page, vk clamped to the last key (the table ends there).
__device__ __forceinline__ uint32_t page_entry(uint32_t vk, uint32_t kv_last, int shift)
{
    return min(vk, kv_last) >> shift << 2;
}

// ---- row outputs and merging --------------------------------------------------------------------
// A q row's O scale: 1 / l; 0 with no keys (l = 0); NaN for non-finite scores.
__device__ __forceinline__ float inv_row_sum(float l, bool nan_row)
{
    return nan_row ? opaque_nan() : l > 0.0f ? __builtin_amdgcn_rcpf(l) : 0.0f;
}

// A q row's LSE as aiter's, ln sum exp(softmax_scale q.k), from frame m (log2 units) and row sum l;
// -inf with no keys, NaN for non-finite scores. l sums the bf16 P: within ln(1 + 2^-8) of exact.
__device__ __forceinline__ float row_lse(float m, float l, bool nan_row)
{
    return nan_row ? opaque_nan() : l > 0.0f ? (m + __builtin_amdgcn_logf(l)) / Log2e : opaque_neg_inf();
}

// A wave's LSEs: its head's at q = 0 (nullptr: none requested) and their token stride; pinned SGPRs.
struct LseRows { float* rows; uint32_t token; };
__device__ __forceinline__ LseRows lse_rows(const Args& a, const WaveWork& w)
{
    const LseRows r{a.lse ? qo_rows(a.lse, a.lse_str, w) : nullptr, (uint32_t)a.lse_str.token};
    pin_scalar(r.rows), pin_scalar(r.token);
    return r;
}
// The LSE of row q, from frame m and row sum l.
__device__ __forceinline__ void store_lse(const LseRows& r, const WaveWork& w, int q, float m, float l)
{
    if (r.rows && (unsigned)q < (unsigned)w.sq)
        r.rows[(size_t)(unsigned)q * r.token] = row_lse(m, l, nan_or_plus_inf(maybe_nan(l)));
}

// Merge a q row's fp32 partials in part order: M = max m, w = exp2(m - M), O = sum w o / sum w l, LSE from
// M and sum w l. A merge item takes merge_ctas CTAs in grid y; a thread owns 8 d of a row.
constexpr int MergeThreads = 256;
constexpr int merge_ctas(int d, int rows) { return NumWaves * rows * (d / 8) / MergeThreads; }
template <int D, int Rows>
__global__ __launch_bounds__(MergeThreads) void merge_parts(Args a)
{
    // 1. This thread's row r of the q block; none past the sequence.
    constexpr int ThreadsPerRow = D / 8;
    const WorkItem it = a.work[blockIdx.x];
    const int r = blockIdx.y * (MergeThreads / ThreadsPerRow) + threadIdx.x / ThreadsPerRow;
    const int d = (threadIdx.x % ThreadsPerRow) * 8;
    const ChunkRows c = chunk_rows(a, it, r / Rows, Rows);
    const int q = c.row0 + r % Rows;
    if ((unsigned)q >= (unsigned)it.sq) return;

    // 2. M over parts with keys (l > 0); a part with l = NaN makes the row NaN.
    const size_t row0 = (size_t)it.slot * BlockM + r;      // part p sits at row0 + p BlockM
    float m_max = NegInf;
    bool nan_row = false;
    for (int p = 0; p < it.num_parts; ++p) {
        const float2 ml = a.part_ml[row0 + p * BlockM];
        if (ml.y > 0.0f) m_max = fmaxf(m_max, ml.x);
        nan_row |= nan_or_plus_inf(maybe_nan(ml.y));
    }

    // 3. O from the weighted partials.
    float l = 0.0f;
    f32x4_t lo{}, hi{};
    for (int p = 0; p < it.num_parts; ++p) {
        const float2 ml = a.part_ml[row0 + p * BlockM];
        const float w = ml.y > 0.0f ? __builtin_amdgcn_exp2f(ml.x - m_max) : 0.0f;
        const f32x4_t* po = reinterpret_cast<const f32x4_t*>(a.part_o + (row0 + p * BlockM) * D + d);
        l += w * ml.y;
        lo += w * po[0];
        hi += w * po[1];
    }
    if (a.lse && d == 0) a.lse[offset(a.lse_str, it.b, it.q0 + q, c.head)] = row_lse(m_max, l, nan_row);
    const float inv = inv_row_sum(l, nan_row);
    bf16x8_t out;
    for (int e = 0; e < 4; ++e) {
        out[e]     = (bf16_t)(lo[e] * inv);
        out[4 + e] = (bf16_t)(hi[e] * inv);
    }
    *reinterpret_cast<bf16x8_t*>(a.o + offset(a.o_str, it.b, it.q0 + q, c.head) + d) = out;
}

// Copy a staged list to its device buffer (see "host: cached lists").
template <class T>
__global__ void copy_list(T* dst, const T* src, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

// ---- device varlen: building the list on the GPU ------------------------------------------------
// Dev mode (cu_seqlens on the GPU). Each q block runs in min(P, its tiles) parts (P from the host), heaviest
// first (counting sort on part length), a q block's CTAs together (head pack, then part); the rest is empty.
// A split part's slot is its item index; merge items follow the list's cap items. Tokens are clamped to the
// tensors' and q rows to max_q (a longer sequence's first q blocks are skipped), so any cu_seqlens stays in
// bounds. Only the list order is nondeterministic.
constexpr int ListThreads = 1024;               // one CTA, one thread per bucket
struct ListArgs {
    const int* cu_q;                            // sequence s: Q/O tokens cu_q[s] .. cu_q[s + 1] - 1 (device)
    const int* cu_k;                            // and K/V tokens cu_k[s] .. cu_k[s + 1] - 1
    int n, max_q, total_q, total_k;
    int rows, pack, packs, causal;
    int parts;                                  // P
    int shift;                                  // bucket of a part: its tiles >> shift
    int cap, cap_merges;                        // items the list can hold, then merge items (P > 1)
    WorkItem* list;
};

template <int Threads>
__global__ __launch_bounds__(Threads) void list_varlen(ListArgs a)
{
    __shared__ int first[Threads];              // per bucket: its item count, then its first slot
    __shared__ int merges;
    const int t = threadIdx.x, lane = t % WaveSize;
    const int block_items = a.parts * a.packs;                        // max items per q block
    const int seq_blocks = num_q_blocks(a.max_q, a.pack, a.rows);     // max q blocks per sequence
    const int num_blocks = a.n * seq_blocks;
    auto block = [&](int v) {                   // q block v % seq_blocks (from the end) of sequence
        if (v >= num_blocks) return WorkItem{}; // v / seq_blocks; an empty item beyond the last q block
        const int s = v / seq_blocks, q0 = min(max(a.cu_q[s], 0), a.total_q);
        const int kv0 = min(max(a.cu_k[s], 0), a.total_k);
        const int sq = min(a.cu_q[s + 1], a.total_q) - q0, skv = max(min(a.cu_k[s + 1], a.total_k) - kv0, 0);
        const int b = (sq > 0 ? num_q_blocks(sq, a.pack, a.rows) : 0) - 1 - v % seq_blocks;
        if (b < 0) return WorkItem{};
        const int tiles = block_tiles(sq, skv, b, a.pack, a.rows, a.causal);
        return WorkItem{q0, sq, kv0, skv, first_chunk(b, sq, a.pack, a.rows), 0, tiles, 0, 0, -1,
                        max(1, min(a.parts, tiles))};
    };
    auto bucket = [&](const WorkItem& it) {     // ordered by part length, longest first
        return Threads - 1 - min(ceil_div(it.tile_hi, it.num_parts) >> a.shift, Threads - 1);
    };

    // 1. Items per bucket, then an exclusive scan: each bucket's first slot.
    first[t] = 0;
    if (t == 0) merges = 0;
    __syncthreads();
    for (int v = t; v < num_blocks; v += Threads) {
        const WorkItem it = block(v);
        if (it.sq > 0) atomicAdd(&first[bucket(it)], it.num_parts * a.packs);
    }
    __syncthreads();
    const int own = first[t];
    int sum = own;
    for (int d = 1; d < Threads; d *= 2) {
        __syncthreads();
        const int add = t >= d ? first[t - d] : 0;
        __syncthreads();
        first[t] = sum += add;
    }
    __syncthreads();
    const int total = first[Threads - 1];
    __syncthreads();
    first[t] = sum - own;
    __syncthreads();

    // 2. Each lane's q block claims slots in its bucket; the wave stores the items, item e (lane j's head pack
    // h, part `part`) by lane e % WaveSize. Merge items go after the list, one lane each.
    for (int v0 = t / WaveSize; v0 < num_blocks; v0 += Threads) {    // lane i handles wave v0 + i
        WorkItem it = block(v0 + lane * (Threads / WaveSize));
        const int lanes = WaveSize - __clzll(__ballot(it.sq > 0));    // through the last lane with a q block
        if (lanes == 0) continue;
        const int n = it.num_parts * a.packs;
        const int to = it.sq > 0 ? atomicAdd(&first[bucket(it)], n) : a.cap;
        for (int e0 = 0; e0 < lanes * block_items; e0 += WaveSize) {  // uniform: shfl reads active lanes only
            const int e = e0 + lane, j = e / block_items % WaveSize;
            const int h = e % block_items / a.parts, part = e % a.parts;
            const int np = max(__shfl(it.num_parts, j), 1), start = __shfl(to, j), at = start + h * np + part;
            int lo = 0, hi = __shfl(it.tile_hi, j);
            if (a.parts > 1) lo = part_lo(part, np, hi), hi = lo + part_len(part, np, hi);
            const WorkItem x{__shfl(it.q0, j), __shfl(it.sq, j), __shfl(it.kv0, j), __shfl(it.skv, j),
                             __shfl(it.chunk0, j), lo, hi, h * a.pack, 0, np > 1 ? at : -1, np};
            const bool fits = start + np * a.packs <= a.cap;         // else cu_seqlens is out of bounds: empty
            if (e < lanes * block_items && part < np && at < a.cap) a.list[at] = fits ? x : WorkItem{};
        }
        if (it.num_parts > 1 && to + n <= a.cap) {
            const int m = atomicAdd(&merges, a.packs);
            for (int h = 0; h < a.packs; ++h)
                it.head0 = h * a.pack, it.slot = to + h * it.num_parts, a.list[a.cap + m + h] = it;
        }
    }

    // 3. Empty the unused items.
    __syncthreads();
    for (int i = min(total, a.cap) + t; i < a.cap; i += Threads) a.list[i] = WorkItem{};
    for (int i = merges + t; i < a.cap_merges; i += Threads) a.list[a.cap + i] = WorkItem{};
}

namespace host {

// plan -> list -> launch: one fwd launch, then a merge launch if any q block splits.
//   host mode  Host lengths: the planner picks wave rows, KV split and CTA order; the list is cached per stream.
//   dev mode   Device cu_seqlens: wave rows and parts P from max lengths and totals; list_varlen builds the
//              list on the GPU (no host sync).
//   plan / run A varlen call planned once per forward pass, run by every layer.

// ---- host: the planner --------------------------------------------------------------------------
// CTAs go round-robin to the 8 XCDs (one L2 each); an XCD starts its CTAs in list order on the first free CU.
// A round: one CTA per CU. A unit: one sequence's head pack (one KV head). The rules:
//   split  The fastest of: q blocks whole, or split to fill k = 1 .. SplitMaxRounds rounds (grids that fit
//          them); 48-row waves, then 32. A later candidate must win by SplitMinGain.
//   fill   Per round, the q block with the longest parts gains a part (at most MaxParts, one tile each).
//   align  At AlignRounds or more rounds, an n-way split cuts at multiples of the longest part w if every
//          last part keeps w / 2: a head's parts run in step and share K/V in L2.
//   time   Modelled dispatch, longest first on the first free CU. A CTA costs its KV tiles (fewer wave rows:
//          q2_tile_cost) plus CtaTiles (a part: PartTiles); a part adds MergeTiles of merge over all CUs.
//          A part anywhere, or LSEs, runs every CTA on the Split=true kernels: tiles cost split_tile_cost.
//   order  Longest first. If the list's eighths carry equal work, XCD x takes eighth x, in groups of the
//          units whose K/V fits LptBytes (at most MaxLptUnits; a split grid is one group); else round-robin.
//   dev    Dev mode: wave rows and P by split's rule on a stand-in call; P > 1 only within DevSplitRounds
//          rounds (bounds the workspace).
constexpr int NumXcd          = 8;
constexpr int SplitMaxRounds  = 8;
constexpr double SplitMinGain = 1.05;
constexpr int MaxParts        = 8;
constexpr double CtaTiles     = 2;
constexpr double PartTiles    = 8;
constexpr double MergeTiles   = 8;
constexpr double LptBytes     = 12 << 20;  // combined L2 of 3 XCDs
constexpr int MaxLptUnits     = 8;
constexpr int DevSplitRounds  = 4;
constexpr int AlignRounds     = 3;

// ---- host: the problem --------------------------------------------------------------------------
// A sequence in batch b: Q/O tokens q0 .. q0 + sq - 1 and K/V tokens kv0 .. kv0 + skv - 1.
struct Seq { int b, q0, sq, kv0, skv; };

struct Problem {
    std::vector<Seq> seqs;
    int H, HKv, causal;
    int pack;                       // q heads per head pack
    int kv_bytes;                   // K + V bytes per key of one KV head
    float q2_tile_cost = 1.0f;      // a KV tile of the fewer wave rows, in tiles of the more
    int rows = MaxWaveRows;         // wave rows, chosen by plan_split
    float split_tile_cost[2] = {1.0f, 1.0f};    // a KV tile on the Split=true kernels, by rows / 16 - 2
    bool lse = false;               // the call returns LSEs
};

// For every sequence and q block: its KV tiles, or its parts.
using PerBlock = std::vector<std::vector<int>>;
inline PerBlock all_tiles(const Problem& p)
{
    PerBlock t(p.seqs.size());
    for (size_t s = 0; s < p.seqs.size(); ++s) {
        const Seq& x = p.seqs[s];
        for (int b = 0; b < num_q_blocks(x.sq, p.pack, p.rows); ++b)
            t[s].push_back(block_tiles(x.sq, x.skv, b, p.pack, p.rows, p.causal));
    }
    return t;
}

// ---- host: 1. wave rows and KV split -----------------------------------------------------------
// A grid's modelled time: CTA dispatch, then the merge; past `limit` only a lower bound.
inline double modelled_time(const Problem& p, const PerBlock& tiles, const PerBlock& parts, int n_cu,
                            double limit)
{
    // 1. The CTAs as (cost, count), plus the merge cost; the bound is the longest CTA or the mean.
    bool split = p.lse;
    for (const auto& s : parts) split = split || std::any_of(s.begin(), s.end(), [](int n) { return n > 1; });
    const double tile = (p.rows == MaxWaveRows ? 1.0 : p.q2_tile_cost) *
                        (split ? p.split_tile_cost[p.rows / 16 - 2] : 1.0);
    const int packs = p.H / p.pack;
    std::vector<std::pair<double, long long>> ctas;
    double merge = 0, work = 0, longest = 0;
    for (size_t s = 0; s < tiles.size(); ++s)
        for (size_t b = 0; b < tiles[s].size(); ++b) {
            const int t = tiles[s][b], n = parts[s][b], longer = t % n;    // parts owning t / n + 1 tiles
            const double fixed = n > 1 ? PartTiles : CtaTiles;
            if (longer > 0) ctas.push_back({(t / n + 1) * tile + fixed, longer * packs});
            ctas.push_back({t / n * tile + fixed, (n - longer) * packs});
            if (n > 1) merge += MergeTiles * n * packs / n_cu;
        }
    for (auto [cost, n] : ctas) longest = std::max(longest, cost), work += cost * n;
    const int merge_time = (int)merge;              // truncated, as the per-CTA costs are whole units
    double end = std::max(longest, work / n_cu);
    if (end + merge_time > limit) return end + merge_time;

    // 2. Within `limit`: longest first, each on the first free CU.
    std::sort(ctas.begin(), ctas.end(), std::greater<>());
    std::vector<std::pair<double, long long>> cu{{0.0, n_cu}};     // levels of (finish time, CUs), a min-heap
    for (auto [cost, n] : ctas)
        while (n > 0 && end + merge_time <= limit) {
            const auto [t, k] = cu[0];
            const long long m = std::min(n, k);
            if ((cu[0].second -= m) == 0)
                std::pop_heap(cu.begin(), cu.end(), std::greater<>()), cu.pop_back();
            cu.push_back({t + cost, m}), std::push_heap(cu.begin(), cu.end(), std::greater<>());
            n -= m, end = std::max(end, t + cost);
        }
    return end + merge_time;
}

// A grid's wave rows (written into p) and its parts for each q block.
inline PerBlock plan_split(Problem& p, int n_cu, bool allow_split)
{
    // 1. Per wave rows (more, then fewer): tiles, parts (one per q block), CTAs, q blocks by part length.
    const int packs = p.H / p.pack, rows[2] = {MaxWaveRows, 32};
    PerBlock tiles[2], parts[2];
    long long ctas[2] = {};
    std::priority_queue<std::tuple<double, int, int>> longest[2];      // per q block: (part length, s, b)
    for (int r = 0; r < 2; ++r) {
        p.rows = rows[r], tiles[r] = all_tiles(p);
        for (const auto& t : tiles[r])
            parts[r].emplace_back(t.size(), 1), ctas[r] += (long long)t.size() * packs;
        if (n_cu <= 0) return parts[0];
        for (size_t s = 0; s < tiles[r].size(); ++s)
            for (size_t b = 0; b < tiles[r][s].size(); ++b) longest[r].push({tiles[r][s][b], (int)s, (int)b});
    }

    // 2. The candidates (split, fill); one with parts unchanged since k - 1 is skipped.
    double best_time = 1e300;
    int best = 0;
    PerBlock best_parts;
    for (int k = 0; k <= (allow_split ? SplitMaxRounds : 0); ++k)
        for (int r = 0; r < 2; ++r) {
            const long long before = ctas[r];
            while (ctas[r] + packs <= (long long)k * n_cu && !longest[r].empty()) {
                const auto [len, s, b] = longest[r].top();
                if (parts[r][s][b] == std::min(MaxParts, tiles[r][s][b])) break;    // at its maximum
                longest[r].pop(), ctas[r] += packs;
                longest[r].push({(double)tiles[r][s][b] / ++parts[r][s][b], s, b});
            }
            if (k > 0 && ctas[r] == before) continue;
            p.rows = rows[r];
            const double time = modelled_time(p, tiles[r], parts[r], n_cu, best_time / SplitMinGain);
            if (SplitMinGain * time <= best_time) best_parts = parts[r], best_time = time, best = r;
        }
    p.rows = rows[best];
    return best_parts;
}

// ---- host: 2. ordering the CTAs -----------------------------------------------------------------
// A CTA: part `part` of `parts` of q block `block` in a unit; len tiles from lo.
struct Cta { int unit, seq, head0, block, lo, len, part, parts, slot; };      // slot: -1, or its partials'

// Unit by unit, q blocks last to first; a split q block's parts consecutive.
inline std::vector<Cta> list_ctas(const Problem& p, const PerBlock& tiles, const PerBlock& parts, int n_cu)
{
    // 1. Per part count n, the longest part and fewest tiles (align); then each part's first tile.
    std::vector<int> width(MaxParts + 1), fewest(MaxParts + 1, INT_MAX);
    long long ctas = 0;
    for (size_t s = 0; s < tiles.size(); ++s)
        for (size_t b = 0; b < tiles[s].size(); ++b) {
            const int n = parts[s][b], t = tiles[s][b];
            width[n] = std::max(width[n], ceil_div(t, n)), fewest[n] = std::min(fewest[n], t);
            ctas += n * (p.H / p.pack);
        }
    auto lo = [&](int j, int n, int t) {
        const bool align = ctas >= (long long)AlignRounds * n_cu && 2 * fewest[n] >= (2 * n - 1) * width[n];
        return align ? std::min(j * width[n], t) : part_lo(j, n, t);
    };

    // 2. The CTAs.
    std::vector<Cta> list;
    int units = 0, slots = 0;
    for (int s = 0; s < (int)tiles.size(); ++s)
        for (int h = 0; h < p.H; h += p.pack, ++units)
            for (int b = (int)tiles[s].size() - 1; b >= 0; --b) {
                const int n = parts[s][b], t = tiles[s][b];
                for (int j = 0; j < n; ++j)
                    list.push_back({units, s, h, b, lo(j, n, t), lo(j + 1, n, t) - lo(j, n, t), j, n,
                                    n > 1 ? slots + j : -1});
                if (n > 1) slots += n;
            }
    return list;
}

// Longest CTAs first; stable counting sort.
inline void longest_first(std::vector<Cta>::iterator first, std::vector<Cta>::iterator last)
{
    const std::vector<Cta> in(first, last);
    std::vector<int> end;                           // per length: CTAs at least that long
    for (const Cta& c : in) end.resize(std::max(end.size(), c.len + size_t(1))), ++end[c.len];
    for (int l = (int)end.size() - 2; l >= 0; --l) end[l] += end[l + 1];
    for (auto c = in.rbegin(); c != in.rend(); ++c) first[--end[c->len]] = *c;
}

// The CTAs in launch order: CTA i runs item i.
inline std::vector<Cta> launch_order(const Problem& p, std::vector<Cta> list)
{
    // 1. Whether the list's eighths carry equal KV tiles, within a tile per CTA and 1%.
    const int n = (int)list.size();
    long long lo = LLONG_MAX, hi = 0;
    for (int x = 0; x < NumXcd; ++x) {
        long long w = 0;
        for (int i = x * n / NumXcd; i < (x + 1) * n / NumXcd; ++i) w += list[i].len;
        lo = std::min(lo, w), hi = std::max(hi, w);
    }
    const bool ranges = n > 0 && n % NumXcd == 0 && hi - lo <= n / NumXcd + hi / 100;

    // 2. Longest first: per eighth when equal, in groups of at least `fit` units (split grid: one group);
    // else the whole list.
    int max_skv = 1;
    for (const Seq& s : p.seqs) max_skv = std::max(max_skv, s.skv);
    const bool split = std::any_of(list.begin(), list.end(), [](const Cta& c) { return c.parts > 1; });
    const int fit = std::clamp((int)(LptBytes / ((double)max_skv * p.kv_bytes)), 1, MaxLptUnits);
    for (int x = 0, m = ranges ? NumXcd : 1; x < m && n > 0; ++x) {
        const auto first = list.begin() + x * n / m, last = list.begin() + (x + 1) * n / m;
        const int u0 = first->unit, units = (last - 1)->unit - u0 + 1;
        const int groups = split || !ranges ? 1 : std::max(1, units / fit);
        auto group = [&](const Cta& c) { return (c.unit - u0) * groups / units; };
        for (auto g = first, run = g; g != last; g = run) {
            run = std::find_if(g, last, [&](const Cta& c) { return group(c) != group(*g); });
            longest_first(g, run);
        }
    }
    if (!ranges) return list;

    // 3. XCD x takes eighth x: CTA i (XCD i % 8) runs item i / 8 of eighth i % 8.
    std::vector<Cta> order(n);
    for (int i = 0; i < n; ++i) order[i] = list[i % NumXcd * (n / NumXcd) + i / NumXcd];
    return order;
}

// ---- host: 3. building the work list ------------------------------------------------------------
// A launch of `ctas` CTAs, then a merge launch if some q block is split.
struct Plan {
    std::vector<WorkItem> list;         // CTA items in launch order, then one per split q block
    int ctas = 0, merges = 0;
    int rows = MaxWaveRows;             // wave rows
    int slots = 0;                      // partial slots
    int parts = 1;                      // dev mode: P
};

// A call's plan (host mode).
inline Plan plan(Problem p, int n_cu, bool allow_split = true)
{
    // 1. Wave rows and parts, then the CTA order.
    auto empty = [](const Seq& s) { return s.sq == 0; };
    p.seqs.erase(std::remove_if(p.seqs.begin(), p.seqs.end(), empty), p.seqs.end());
    const PerBlock parts = plan_split(p, n_cu, allow_split);
    const std::vector<Cta> order = launch_order(p, list_ctas(p, all_tiles(p), parts, n_cu));

    // 2. The items.
    auto item = [&](const Cta& c) {
        const Seq& s = p.seqs[c.seq];
        return WorkItem{s.q0, s.sq, s.kv0, s.skv, first_chunk(c.block, s.sq, p.pack, p.rows), c.lo,
                        c.lo + c.len, c.head0, s.b, c.slot, c.parts};
    };
    Plan plan;
    plan.list.reserve(order.size() * 2);
    for (const Cta& c : order) plan.list.push_back(item(c)), plan.slots += c.slot >= 0;
    for (const Cta& c : order)
        if (c.parts > 1 && c.part == 0) plan.list.push_back(item(c)), ++plan.merges;
    plan.ctas = (int)order.size();
    plan.rows = p.rows;
    return plan;
}

// ---- host: 4. dev mode --------------------------------------------------------------------------
// Sequences of at most mq q rows and mk keys; tensors of total_q and total_k tokens. Wave rows and P by
// split's rule on a stand-in call (the longest sequence, the rest even). The list holds the most q blocks
// these allow, times P; ctas = 0 when too big.
inline Plan dev_plan(Problem p, int n, int mq, int mk, int total_q, int total_k, int n_cu)
{
    // 1. The stand-in call, and the most q blocks mq and the totals allow.
    const int packs = p.H / p.pack;
    p.seqs.assign(1, {0, 0, mq, 0, mk});
    if (n > 1)
        p.seqs.insert(p.seqs.end(), n - 1, {0, 0, (total_q - mq) / (n - 1), 0, (total_k - mk) / (n - 1)});
    auto max_blocks = [&](int rows) {
        const long long chunks = std::min((long long)n * head_chunks(mq, rows),
                                          ((long long)total_q + (long long)n * (rows - 1)) / rows);
        return std::min((long long)n * num_q_blocks(mq, p.pack, rows),
                        (p.pack * chunks + (NumWaves - 1LL) * n) / NumWaves);
    };

    // 2. Wave rows and P: fastest modelled time; a later candidate must win by SplitMinGain.
    int rows = MaxWaveRows, parts = 1;
    double best = 1e300;
    for (int P = 1; P <= MaxParts && max_blocks(MaxWaveRows) * packs <= SplitMaxRounds * n_cu; ++P)
        for (int r : {MaxWaveRows, 32}) {
            if (P > 1 && max_blocks(r) * packs * P > DevSplitRounds * n_cu) continue;
            p.rows = r;
            const PerBlock tiles = all_tiles(p);
            PerBlock np = tiles;
            for (auto& s : np) for (int& x : s) x = std::max(1, std::min(P, x));
            const double time = modelled_time(p, tiles, np, n_cu, best / SplitMinGain);
            if (SplitMinGain * time <= best) best = time, rows = r, parts = P;
        }

    // 3. The list: max_blocks q blocks of P parts per head pack, then a merge item each (P > 1).
    Plan plan;
    const long long items = max_blocks(rows) * packs;
    if (items * (parts + 1) > INT_MAX / (int)sizeof(WorkItem) ||
        (long long)n * num_q_blocks(mq, p.pack, rows) > INT_MAX)
        return plan;
    plan.rows = rows;
    plan.ctas = (int)(items * parts);
    plan.merges = parts > 1 ? (int)items : 0;
    plan.slots = parts > 1 ? plan.ctas : 0;
    plan.parts = parts;
    return plan;
}

// ---- host: cached lists -------------------------------------------------------------------------
// Each stream caches its last StagingSlots plans and device lists, keyed by head dims, heads and sequences
// (repeated layer calls skip planning and copying), and a partials workspace. A new list goes through a
// pinned buffer and a copy_list kernel, in stream order. No call blocks: nothing is freed before release
// (hipFree waits for the device). state_mutex guards host state; hipStreamPerThread is keyed per thread.
// Under capture a list is uploaded once and kept (launch_captured). Graphs sharing a stream's workspace
// must not replay concurrently.
constexpr int MaxDevices    = 64;
constexpr int StagingSlots  = 8;
constexpr int PinnedBuffers = 1024; // per device; when all are busy, a copy waits for the oldest

// A device buffer that only grows (at least doubling); outgrown ones are kept until release.
struct Buffer { void* ptr = nullptr; size_t bytes = 0; std::vector<void*> kept; };

struct Staged {
    std::vector<int> key;           // empty means not reusable
    Plan plan;
    Buffer list;                    // the device list
};

// A list's pinned copy source; done is recorded after its last copy.
struct Pinned { WorkItem* ptr = nullptr; size_t cap = 0; hipEvent_t done = nullptr; };

struct StreamState {
    Staged staged[StagingSlots];    // a ring buffer
    int next = 0;
    Buffer ws;
};

inline std::mutex state_mutex;
inline int device_cus[MaxDevices];
inline std::unordered_map<hipStream_t, StreamState> streams[MaxDevices];
inline std::deque<Pinned> pinned[MaxDevices];                           // oldest copy first; never freed
inline std::map<std::vector<int>, WorkItem*> graph_lists[MaxDevices];     // launch_captured's lists, by content
inline hipStream_t upload_streams[MaxDevices];                           // the ones device_alloc uses

// The current device, or -1 (on error, or beyond MaxDevices).
inline int current_device()
{
    int dev = 0;
    return hipGetDevice(&dev) == hipSuccess && dev >= 0 && dev < MaxDevices ? dev : -1;
}

// The key for a stream's state: hipStreamPerThread keys on this thread, not on the shared handle value.
inline hipStream_t state_key(hipStream_t stream)
{
    thread_local const char thread_stream = 0;     // its address stands for this thread's own stream
    return stream == hipStreamPerThread ? (hipStream_t)&thread_stream : stream;
}

inline int cu_count(int dev)
{
    int& n = device_cus[dev];
    if (n == 0 && hipDeviceGetAttribute(&n, hipDeviceAttributeMultiprocessorCount, dev) != hipSuccess) n = -1;
    return n;
}

// The stream's graph capture id, 0 when there is none (capture ids start at 1).
inline unsigned long long capture_id(hipStream_t stream)
{
    hipStreamCaptureStatus s = hipStreamCaptureStatusNone;
    unsigned long long id = 0;
    (void)hipStreamGetCaptureInfo(stream, &s, &id);
    return s == hipStreamCaptureStatusNone ? 0 : id;
}
inline bool capturing(hipStream_t stream) { return capture_id(stream) != 0; }

// A pinned buffer for n items: the pool's oldest once its copy is done (waited for when all PinnedBuffers
// are busy), else a new one; power-of-two sized, outgrown ones dropped unfreed. False when out of memory.
inline bool take_pinned(Pinned& p, int dev, size_t n)
{
    std::deque<Pinned>& pool = pinned[dev];
    if (!pool.empty() && (pool.size() >= PinnedBuffers ? hipEventSynchronize(pool.front().done)
                                                       : hipEventQuery(pool.front().done)) == hipSuccess)
        p = pool.front(), pool.pop_front();
    if (!p.done && hipEventCreateWithFlags(&p.done, hipEventDisableTiming) != hipSuccess) return false;
    if (p.cap >= n) return true;
    size_t cap = 1024;
    while (cap < n) cap *= 2;
    void* ptr = nullptr;
    if (hipHostMalloc(&ptr, cap * sizeof(WorkItem)) != hipSuccess) return pool.push_back(p), false;
    p.ptr = (WorkItem*)ptr, p.cap = cap;
    return true;
}

// Copy a list into device memory `to` in stream order, via a pinned buffer. Returns false on OOM.
inline bool upload(void* to, const std::vector<WorkItem>& list, int dev, hipStream_t stream)
{
    const int n = (int)list.size();
    Pinned p;
    if (!to || !take_pinned(p, dev, n)) return false;
    std::memcpy(p.ptr, list.data(), n * sizeof(WorkItem));
    const dim3 grid(ceil_div(n, 256));
    hipLaunchKernelGGL(copy_list<WorkItem>, grid, dim3(256), 0, stream, (WorkItem*)to, p.ptr, n);
    (void)hipEventRecord(p.done, stream);
    pinned[dev].push_back(p);
    return true;
}

// Device memory, filled from init if given; nullptr on failure. Capture-safe: relaxed mode allows
// hipMalloc, and the copy runs on its own stream (not the null stream, which the capture would take).
inline void* device_alloc(size_t bytes, const void* init = nullptr, int dev = 0)
{
    void* p = nullptr;
    hipStreamCaptureMode mode = hipStreamCaptureModeRelaxed;
    (void)hipThreadExchangeStreamCaptureMode(&mode);
    hipStream_t& s = upload_streams[dev];
    if (hipMalloc(&p, bytes) != hipSuccess) p = nullptr;
    else if (init && ((!s && hipStreamCreateWithFlags(&s, hipStreamNonBlocking) != hipSuccess) ||
                      hipMemcpyAsync(p, init, bytes, hipMemcpyHostToDevice, s) != hipSuccess ||
                      hipStreamSynchronize(s) != hipSuccess))
        p = nullptr;                                // leaked: a free could end the capture
    if (!p) (void)hipGetLastError();              // clear it for the next launch
    (void)hipThreadExchangeStreamCaptureMode(&mode);
    return p;
}

// Grow b until it holds at least `bytes`, or return nullptr.
inline void* grow(Buffer& b, size_t bytes)
{
    if (bytes > b.bytes) {
        if (b.ptr) b.kept.push_back(b.ptr);
        bytes = std::max(bytes, 2 * b.bytes);
        b.ptr = device_alloc(bytes), b.bytes = b.ptr ? bytes : 0;
    }
    return b.ptr;
}

// Free b once its kernels are done and no graph will replay them.
inline void release(Buffer& b)
{
    for (void* p : b.kept) (void)hipFree(p);
    if (b.ptr) (void)hipFree(b.ptr);
    b = {};
}

// Stage a plan in slot s, copying its list to the slot's device buffer.
inline Staged* stage(Staged& s, std::vector<int> key, Plan plan, int dev, hipStream_t stream)
{
    s.key.clear();
    if (!upload(grow(s.list, plan.list.size() * sizeof(WorkItem)), plan.list, dev, stream)) return nullptr;
    s.key = std::move(key);
    s.plan = std::move(plan);
    return &s;
}

// Stage a plan in the stream's next slot.
inline Staged* stage(StreamState& st, std::vector<int> key, Plan plan, int dev, hipStream_t stream)
{
    Staged& s = st.staged[st.next];
    st.next = (st.next + 1) % StagingSlots;
    return stage(s, std::move(key), std::move(plan), dev, stream);
}

// ---- host: launches -----------------------------------------------------------------------------
// A head dim's kernels fwd[r][split] for wave rows 32 and 48 (r = rows / 16 - 2; split: plans with parts
// or LSEs), a 32-row wave's KV tile cost in 48-row tiles, and a tile's cost on fwd[r][true] over fwd[r][false].
struct Kernels { void (*fwd[2][2])(Args); float q2_tile_cost; float split_tile_cost[2]; };

// The split workspace: part_o first, then part_ml; a part holds BlockM rows.
template <int Dv>
size_t workspace_bytes(const Plan& plan)
{
    return (size_t)plan.slots * BlockM * (Dv * sizeof(float) + sizeof(float2));
}

// Run a plan: first its CTAs, then its merges.
template <int Dv>
int run_plan(Kernels k, Args a, const Plan& plan, const WorkItem* list, void* ws, hipStream_t stream)
{
    if (plan.ctas == 0) return 0;
    a.work = list;
    a.part_o = (float*)ws;
    a.part_ml = (float2*)((char*)ws + (size_t)plan.slots * BlockM * Dv * sizeof(float));
    const auto fwd = k.fwd[plan.rows / 16 - 2][plan.merges > 0 || a.lse];
    hipLaunchKernelGGL(fwd, dim3(plan.ctas), dim3(BlockSize), 0, stream, a);
    if (plan.merges > 0) {
        a.work = list + plan.ctas;
        const auto merge = plan.rows == MaxWaveRows ? merge_parts<Dv, 48> : merge_parts<Dv, 32>;
        const dim3 grid(plan.merges, merge_ctas(Dv, plan.rows));
        hipLaunchKernelGGL(merge, grid, dim3(MergeThreads), 0, stream, a);
    }
    return (int)hipGetLastError();
}

// A call under graph capture: planned anew, its list uploaded outside the graph and kept (equal lists
// shared), partials in the stream's workspace; replays run only kernels.
template <int Dv>
int launch_captured(Kernels k, Args a, const Problem& p, int dev, StreamState& st, hipStream_t stream)
{
    // 1. The plan and its workspace (none: unsplit).
    const int n_cu = cu_count(dev);
    Plan plan = host::plan(p, n_cu);
    void* ws = nullptr;
    if (plan.slots > 0 && !(ws = grow(st.ws, workspace_bytes<Dv>(plan))))
        plan = host::plan(p, n_cu, false);

    // 2. The list (cached or new), then the kernels.
    const size_t bytes = plan.list.size() * sizeof(WorkItem);
    std::vector<int> key(bytes / sizeof(int));
    std::memcpy(key.data(), plan.list.data(), bytes);
    WorkItem*& list = graph_lists[dev][key];
    if (!list) list = (WorkItem*)device_alloc(bytes, plan.list.data(), dev);
    if (!list) return (int)hipErrorOutOfMemory;
    return run_plan<Dv>(k, a, plan, list, ws, stream);
}

// The Problem for a call, with no sequences yet.
template <int Dqk, int Dv>
Problem problem(Kernels k, int H, int HKv, int causal)
{
    return {{}, H, HKv, causal, causal ? H / HKv : 1, (Dqk + Dv) * (int)sizeof(bf16_t), k.q2_tile_cost,
            MaxWaveRows, {k.split_tile_cost[0], k.split_tile_cost[1]}};
}

// Launch a call (host mode) from the stream's matching slot, or a newly staged plan.
template <int Dqk, int Dv>
int launch(Kernels k, Args a, Problem p, hipStream_t stream)
{
    const int dev = current_device();
    if (dev < 0) return -1;
    p.lse = a.lse != nullptr;
    std::lock_guard<std::mutex> lock(state_mutex);    // until the kernels are queued
    const int n_cu = cu_count(dev);
    StreamState& st = streams[dev][state_key(stream)];
    if (capturing(stream)) return launch_captured<Dv>(k, a, p, dev, st, stream);

    // 1. The plan and its device list.
    std::vector<int> key{Dqk, Dv, p.H, p.HKv, p.causal, p.lse};
    for (const Seq& s : p.seqs) key.insert(key.end(), {s.b, s.q0, s.sq, s.kv0, s.skv});
    const Staged* s = std::find_if(st.staged, st.staged + StagingSlots,
                                   [&](const Staged& x) { return x.key == key; });
    if (s == st.staged + StagingSlots) s = stage(st, key, host::plan(p, n_cu), dev, stream);
    if (!s) return (int)hipErrorOutOfMemory;

    // 2. The split workspace (none: unsplit, from a list staged for this call).
    void* ws = nullptr;
    if (s->plan.slots > 0 && !(ws = grow(st.ws, workspace_bytes<Dv>(s->plan)))) {
        s = stage(st, {}, host::plan(p, n_cu, false), dev, stream);
        if (!s) return (int)hipErrorOutOfMemory;
    }

    // 3. The kernels.
    return run_plan<Dv>(k, a, s->plan, (const WorkItem*)s->list.ptr, ws, stream);
}

// A batch of B sequences, each Sq q rows and Skv keys, one sequence per batch.
template <int Dqk, int Dv>
int launch_batch(Kernels k, Args a, int B, int Sq, int Skv, hipStream_t stream)
{
    if (B == 0 || a.H == 0 || Sq == 0) return 0;
    Problem p = problem<Dqk, Dv>(k, a.H, a.HKv, a.causal);
    for (int b = 0; b < B; ++b) p.seqs.push_back({b, 0, Sq, 0, Skv});
    return launch<Dqk, Dv>(k, a, p, stream);
}

// Varlen sequences from host arrays: sequence i is Q/O tokens cu_q[i] .. cu_q[i + 1] - 1 and K/V tokens
// cu_k[i] .. of batch 0. False when invalid.
inline bool varlen_seqs(Problem& p, const int* cu_q, const int* cu_k, int n)
{
    if (!cu_q || !cu_k || n < 0 || cu_q[0] < 0 || cu_k[0] < 0) return false;
    for (int i = 0; i < n; ++i) {
        const Seq s{0, cu_q[i], cu_q[i + 1] - cu_q[i], cu_k[i], cu_k[i + 1] - cu_k[i]};
        if (s.sq < 0 || s.skv < 0) return false;
        p.seqs.push_back(s);
    }
    return true;
}

// Varlen using host arrays.
template <int Dqk, int Dv>
int launch_varlen(Kernels k, Args a, const int* cu_q, const int* cu_k, int num_seqs, hipStream_t stream)
{
    Problem p = problem<Dqk, Dv>(k, a.H, a.HKv, a.causal);
    if (!varlen_seqs(p, cu_q, cu_k, num_seqs)) return -1;
    if (a.H == 0 || cu_q[num_seqs] == cu_q[0]) return 0;
    return launch<Dqk, Dv>(k, a, p, stream);
}

// ---- host: varlen plans -------------------------------------------------------------------------
// A varlen plan (the C API's handle): host mode given host cu_seqlens outside a capture, else dev mode
// (list_varlen reads cu_seqlens at run time, in a graph at every replay). A run follows its plan in stream
// order, in the same capture or none. The plan owns its buffers until release().
struct VarlenCall {
    const int *cu_q, *cu_k;                 // device pointers
    const int *host_cu_q, *host_cu_k;       // host copies, or null
    int n, max_q, max_k, total_q, total_k;
};

struct VarlenPlan {
    int dqk = 0, dv = 0, H = 0, HKv = 0, causal = 0;   // the call's values, reused by its runs
    bool ready = false;                                // the last plan call succeeded
    unsigned long long capture = 0;                    // its graph capture, 0 for none
    Plan plan;                                         // what run_plan launches, with list
    const WorkItem* list = nullptr;
    void* ws = nullptr;                                // workspace.ptr when the plan has slots
    Buffer host_list, dev_list, workspace;             // each mode's list; the partials
};

inline std::map<std::array<int, 11>, Plan> dev_plans;                    // keyed by head dims, heads, counts, CUs
inline std::unordered_map<hipStream_t, VarlenPlan> stream_plans[MaxDevices];   // launch_varlen_dev's

// Plan h for call c; state_mutex held.
template <int Dqk, int Dv>
int plan_varlen(VarlenPlan& h, Kernels k, int H, int HKv, int causal, const VarlenCall& c, hipStream_t stream)
{
    const int dev = current_device();
    if (dev < 0) return -1;
    h.dqk = Dqk, h.dv = Dv, h.H = H, h.HKv = HKv, h.causal = causal, h.capture = capture_id(stream);
    h.plan = {}, h.list = nullptr, h.ws = nullptr;
    Problem p = problem<Dqk, Dv>(k, H, HKv, causal);
    const int n_cu = cu_count(dev);

    // 1. Host mode: the planner's plan (no workspace: unsplit).
    if (c.host_cu_q && c.host_cu_k && !h.capture) {
        if (!varlen_seqs(p, c.host_cu_q, c.host_cu_k, c.n)) return -1;
        if (H == 0 || c.host_cu_q[c.n] == c.host_cu_q[0]) return 0;
        Plan plan = host::plan(p, n_cu);
        if (plan.slots > 0 && !(h.ws = grow(h.workspace, workspace_bytes<Dv>(plan))))
            plan = host::plan(p, n_cu, false);
        void* list = grow(h.host_list, plan.list.size() * sizeof(WorkItem));
        if (!upload(list, plan.list, dev, stream)) return (int)hipErrorOutOfMemory;
        h.plan = std::move(plan), h.list = (const WorkItem*)list;
        return 0;
    }

    // 2. Dev mode: rows and P from the host numbers (cached), then list_varlen.
    if (!c.cu_q || !c.cu_k || c.n < 0 || c.max_q < 0 || c.max_k < 0 || c.total_q < 0 || c.total_k < 0)
        return -1;
    if (c.n == 0 || H == 0 || c.max_q == 0 || c.total_q == 0) return 0;
    const int mq = std::min(c.max_q, c.total_q), mk = std::min(c.max_k, c.total_k);
    const std::array<int, 11> key{Dqk, Dv, H, HKv, causal, c.n, mq, mk, c.total_q, c.total_k, n_cu};
    if (dev_plans.size() > 256) dev_plans.clear();
    auto it = dev_plans.find(key);
    if (it == dev_plans.end())
        it = dev_plans.emplace(key, dev_plan(p, c.n, mq, mk, c.total_q, c.total_k, n_cu)).first;
    const Plan& plan = it->second;
    if (plan.ctas == 0) return -1;
    int shift = 0;                                     // part lengths spread over ListThreads buckets
    while ((ceil_div(mk, KvTile) >> shift) >= ListThreads) ++shift;
    void* list = grow(h.dev_list, (size_t)(plan.ctas + plan.merges) * sizeof(WorkItem));
    void* ws = plan.slots > 0 ? grow(h.workspace, workspace_bytes<Dv>(plan)) : nullptr;
    if (!list || (plan.slots > 0 && !ws)) return (int)hipErrorOutOfMemory;
    const ListArgs la{c.cu_q, c.cu_k, c.n, mq, c.total_q, c.total_k, plan.rows, p.pack, H / p.pack, causal,
                      plan.parts, shift, plan.ctas, plan.merges, (WorkItem*)list};
    hipLaunchKernelGGL(list_varlen<ListThreads>, dim3(1), dim3(ListThreads), 0, stream, la);
    h.plan = plan, h.list = (const WorkItem*)list, h.ws = ws;
    return (int)hipGetLastError();
}

// Run plan h over tensors a; -1 if h was planned for other head dims or capture state.
template <int Dqk, int Dv>
int run_varlen(const VarlenPlan& h, Kernels k, const Args& a, hipStream_t stream)
{
    if (h.dqk != Dqk || h.dv != Dv || capture_id(stream) != h.capture) return -1;
    return run_plan<Dv>(k, a, h.plan, h.list, h.ws, stream);
}

// Free h's buffers once its kernels are done and no graph will replay its runs.
inline void release(VarlenPlan& h)
{
    release(h.host_list), release(h.dev_list), release(h.workspace);
    h = {};
}

// A varlen call on device cu_seqlens, planned and run now with the stream's plan. Host lengths, if given,
// go to launch_varlen outside a capture (replays may change cu).
template <int Dqk, int Dv>
int launch_varlen_dev(Kernels k, const Args& a, VarlenCall c, hipStream_t stream)
{
    if (c.host_cu_q && c.host_cu_k && !capturing(stream))
        return launch_varlen<Dqk, Dv>(k, a, c.host_cu_q, c.host_cu_k, c.n, stream);
    const int dev = current_device();
    if (dev < 0) return -1;
    std::lock_guard<std::mutex> lock(state_mutex);    // until the kernels are queued
    VarlenPlan& h = stream_plans[dev][state_key(stream)];
    c.host_cu_q = c.host_cu_k = nullptr;
    const int err = plan_varlen<Dqk, Dv>(h, k, a.H, a.HKv, a.causal, c, stream);
    return err ? err : run_varlen<Dqk, Dv>(h, k, a, stream);
}

// ---- host: the C API ----------------------------------------------------------------------------
// attention.h's entries forward to api_x with their kernels; -1: invalid arguments.

// A call's tensors as the C API passes them (see "tensors").
struct Tensors {
    const void *q, *k, *v;
    void* o;
    float* lse;                 // nullptr when there are no LSEs
    const int64_t* strides;     // q's batch, token and head strides, then k's, v's, o's (and lse's)
};

// Whether scale is positive and finite, from its bits (-ffast-math assumes finite floats).
inline bool valid_scale(float scale)
{
    const volatile float opaque = scale;
    const uint32_t bits = __builtin_bit_cast(uint32_t, (float)opaque);
    return bits > 0 && bits < 0x7F800000u;         // positive, below +inf
}

// Strides: non-negative, token strides at most MaxToken elements (32-bit in the kernels; a KV tile's bytes
// fit the DMA's 32-bit offsets).
constexpr int64_t MaxToken = 1 << 22;
inline bool valid_strides(const int64_t* s, bool lse)
{
    if (!s) return false;
    for (int i = 0; i < (lse ? 15 : 12); ++i)
        if (s[i] < 0 || (i % 3 == 1 && s[i] > MaxToken)) return false;
    return true;
}

// Valid scale and strides.
inline bool valid_call(const Tensors& t, float scale)
{
    return valid_scale(scale) && valid_strides(t.strides, t.lse);
}
// Valid heads: H a multiple of HKv (MHA, GQA or MQA).
inline bool valid_heads(int H, int HKv) { return H >= 0 && HKv > 0 && H % HKv == 0; }

// Contiguous [batch][token][heads][d] tensors (BSHD; THD when Sq = Skv = 0).
inline std::array<int64_t, 12> contiguous_strides(int Sq, int Skv, int H, int HKv, int Dqk, int Dv)
{
    auto bshd = [](int64_t s, int64_t h, int64_t d) { return std::array<int64_t, 3>{s * h * d, h * d, d}; };
    const auto q = bshd(Sq, H, Dqk), k = bshd(Skv, HKv, Dqk), v = bshd(Skv, HKv, Dv), o = bshd(Sq, H, Dv);
    return {q[0], q[1], q[2], k[0], k[1], k[2], v[0], v[1], v[2], o[0], o[1], o[2]};
}

// Contiguous pools of [pages][page_size][heads][d] (q and o use THD).
inline std::array<int64_t, 12> paged_strides(int page_size, int H, int HKv, int Dqk, int Dv)
{
    auto s = contiguous_strides(0, 0, H, HKv, Dqk, Dv);
    s[3] = (int64_t)page_size * HKv * Dqk, s[6] = (int64_t)page_size * HKv * Dv;
    return s;
}

// A call's Args (the launch sets work, part_o, part_ml). Causal GQA packs a KV head's q heads (they share
// partial q blocks and diagonals); dense calls do not pack.
inline Args make_args(const Tensors& t, int H, int HKv, int causal, float scale)
{
    auto str = [&](int i) { return Strides{t.strides[3 * i], t.strides[3 * i + 1], t.strides[3 * i + 2]}; };
    return {(const bf16_t*)t.q, (const bf16_t*)t.k, (const bf16_t*)t.v, (bf16_t*)t.o, str(0), str(1), str(2),
            str(3), nullptr, nullptr, nullptr, H, HKv, H / HKv, causal ? H / HKv : 1, causal, scale * Log2e,
            t.lse, t.lse ? str(4) : Strides{}};
}

template <int Dqk, int Dv>
int api_fwd(Kernels k, const Tensors& t, int B, int H, int HKv, int Sq, int Skv, int causal, float scale,
            hipStream_t stream)
{
    if (B < 0 || Sq < 0 || Skv < 0 || !valid_heads(H, HKv) || !valid_call(t, scale)) return -1;
    return launch_batch<Dqk, Dv>(k, make_args(t, H, HKv, causal, scale), B, Sq, Skv, stream);
}

template <int Dqk, int Dv>
int api_varlen_fwd(Kernels k, const Tensors& t, const int* cu_q, const int* cu_k, int num_seqs, int H,
                   int HKv, int causal, float scale, hipStream_t stream)
{
    if (num_seqs < 0 || !valid_heads(H, HKv) || !valid_call(t, scale)) return -1;
    return launch_varlen<Dqk, Dv>(k, make_args(t, H, HKv, causal, scale), cu_q, cu_k, num_seqs, stream);
}

template <int Dqk, int Dv>
int api_varlen_fwd_dev(Kernels k, const Tensors& t, const VarlenCall& c, int H, int HKv, int causal,
                       float scale, hipStream_t stream)
{
    if (c.n < 0 || !valid_heads(H, HKv) || !valid_call(t, scale)) return -1;
    return launch_varlen_dev<Dqk, Dv>(k, make_args(t, H, HKv, causal, scale), c, stream);
}

template <int Dqk, int Dv>
int api_varlen_plan(void* plan, Kernels k, const VarlenCall& c, int H, int HKv, int causal,
                    hipStream_t stream)
{
    if (!plan || c.n < 0 || !valid_heads(H, HKv)) return -1;
    std::lock_guard<std::mutex> lock(state_mutex);
    VarlenPlan& h = *(VarlenPlan*)plan;
    h.ready = false;
    const int rc = plan_varlen<Dqk, Dv>(h, k, H, HKv, causal, c, stream);
    h.ready = rc == 0;
    return rc;
}

template <int Dqk, int Dv>
int api_varlen_run(const void* plan, Kernels k, const Tensors& t, float scale, hipStream_t stream)
{
    if (!plan || !valid_call(t, scale)) return -1;
    const VarlenPlan& h = *(const VarlenPlan*)plan;
    if (!h.ready) return -1;                           // unplanned, or its last plan failed
    return run_varlen<Dqk, Dv>(h, k, make_args(t, h.H, h.HKv, h.causal, scale), stream);
}

// ---- host: paged K/V ----------------------------------------------------------------------------
// Sequence i: Q/O tokens cu_q[i] .. cu_q[i + 1] - 1, kv_lens[i] keys from table entry page_starts[i] (host
// arrays). page_size: a power of two, at least a wave's DMA rows (8 for D128, 16 for MLA).
template <int Dqk, int Dv>
int api_paged_fwd(Kernels k, const Tensors& t, const int* cu_q, const int* table, int table_len,
                  const int* page_starts, const int* kv_lens, int n, int page_size, int H, int HKv,
                  int causal, float scale, hipStream_t stream)
{
    // 1. Checks: page and page-row bytes fit 32 bits, rows at least d, K and V pools match (D128 shifts K's
    // resource for V), entries within the table, virtual tokens within 31 bits.
    const int P = page_size;
    if (n < 0 || !valid_heads(H, HKv) || !valid_call(t, scale) || !cu_q || !page_starts || !kv_lens ||
        !table || table_len < 1 || table_len > INT_MAX / 4 || P < (Dqk == Dv ? 8 : 16) || (P & (P - 1)) ||
        cu_q[0] < 0)
        return -1;
    for (int i : {1, 2})                            // k's then v's strides: page, token, head
        if (t.strides[3 * i] * 2 > UINT32_MAX || t.strides[3 * i + 1] < (i == 1 ? Dqk : Dv) ||
            (int64_t)P * t.strides[3 * i + 1] * 2 > UINT32_MAX)
            return -1;
    if (Dqk == Dv && !std::equal(t.strides + 3, t.strides + 6, t.strides + 6)) return -1;
    Problem p = problem<Dqk, Dv>(k, H, HKv, causal);
    for (int i = 0; i < n; ++i) {
        const int start = page_starts[i], len = kv_lens[i], sq = cu_q[i + 1] - cu_q[i];
        if (start < 0 || len < 0 || sq < 0 || ceil_div(len, P) > table_len - start ||
            (int64_t)start * P + len > INT_MAX)
            return -1;
        p.seqs.push_back({0, cu_q[i], sq, start * P, len});
    }
    if (H == 0 || cu_q[n] == cu_q[0]) return 0;

    // 2. A varlen call over the pages.
    Args a = make_args(t, H, HKv, causal, scale);
    a.page_table = table, a.table_len = table_len;
    while ((1 << a.page_shift) < P) ++a.page_shift;
    return launch<Dqk, Dv>(k, a, p, stream);
}

}  // namespace host
}  // namespace attn
