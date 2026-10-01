"""Exercise the MLA absorbed-decode op (mla_decode_a16w8) against an fp32 reference.

The reference dequantizes the SAME fp8 KV the kernel reads, so the only error being probed is the
kernel's own bf16 QK/PV arithmetic rather than the quantization.

Batches are DENSE: every request decodes the same q_len, so q_lat/q_pe/o_lat are [B*q_len, H, *] and
lse is [B*q_len, H], with request b owning rows [b*q_len, (b+1)*q_len).

At cp_world 1 the op is plain decode and speculative verify: q_len 1..8 at the head counts a
tensor-parallel shard runs, plus the end-aligned causal window and the optional LSE.

Under DCP the KV pool is POSITION-sharded across cp_world ranks: the token at global position p
lives only on rank p % cp_world, at pool row p // cp_world. Q is head-replicated, so a single rank
runs ALL H <= 128 heads over its 1/N of the positions and returns a rank-local partial along with
the base-2 LSE that weights it in the cross-rank merge. Covered across ranks, draft lengths and the
H domain (including values that do not divide the 96-row CTA slice), rank-local windows that are
EMPTY, and the merge (mla_dcp_lse_merge_ranks) that follows the DCP all-to-all.

Both paths are additionally run under CUDA-graph capture.
"""

import pytest
import torch

import moonmath_amd as ma
from moonmath_amd import _arch

# gfx942-only kernels: skip when this build lacks the module or the GPU is not CDNA3.
pytestmark = pytest.mark.skipif(
    not _arch.is_built("gfx942") or (torch.cuda.is_available() and _arch.get_gfx_runtime() != "gfx942"),
    reason="needs moonmath_amd._C_gfx942 and a gfx942 GPU",
)

LAT, ROPE = 512, 64
FUSED = LAT + ROPE  # = 576
SCALE = 0.088  # 1/sqrt(qk_head_dim) times the YaRN mscale, matching the model
KV_SCALE = 0.02
LOG2E = 1.4426950408889634
O_ABS = 0.02  # bound on |o - ref|: bf16 QK/PV vs an fp32 reference over the same dequantized KV
LSE_ABS = 0.05  # bound on |lse - ref| in base-2
MERGE_ABS = 0.01  # a pure fp32 reduction, so the cross-rank merge is held tighter


@pytest.fixture(scope="session")
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA/HIP device not available")
    return torch.device("cuda")


FP8_MAX = torch.finfo(torch.float8_e4m3fnuz).max      # 240 -- FNUZ's larger exponent bias trades
                                                      # top-end range for bottom-end range, so it
                                                      # is NOT e4m3-FN's 448.


def _quantize_fp8(x):
    """Quantize to fp8 e4m3-fnuz using the per-tensor KV_SCALE."""
    return torch.clamp(x / KV_SCALE, -FP8_MAX, FP8_MAX).to(torch.float8_e4m3fnuz)


def _local_positions(count, cp_world, cp_rank, device):
    """Global positions below `count` owned by rank cp_rank: those with p % cp_world == cp_rank.

    Empty when the request is shorter than the rank's own offset -- which really happens at
    cp_world 8 with a 3-token request, and is the case the empty-window contract targets.
    """
    return torch.arange(cp_rank, max(count, cp_rank), cp_world, device=device)


def _build_case(B, q_len, H, cp_world, cp_rank, glen, device, seed):
    """Position-shard one random dense batch onto rank `cp_rank` of `cp_world`.

    `glen` is either the per-request list of GLOBAL KV lengths, or an upper bound to sample them
    from. Rank-local slots are a random permutation of the pool, so a kernel that assumed
    contiguity would fail. Returns the kernel's tensors along with `kv_deq`, the fp32
    dequantization of the SAME fp8 rows indexed by GLOBAL position for the reference. At
    cp_world 1 the rank holds every position.
    """
    torch.manual_seed(seed)

    if isinstance(glen, (list, tuple)):
        g_len = torch.tensor(glen, dtype=torch.int32, device=device)
    else:
        g_len = torch.randint(max(q_len, glen // 2), glen + 1, (B,), dtype=torch.int32, device=device)

    # rank cp_rank holds the positions p < glen[b] where p % cp_world == cp_rank
    s_loc = (g_len // cp_world + (cp_rank < g_len % cp_world).int()).int()
    g_max = int(g_len.max())

    kv_q = _quantize_fp8(torch.randn(B, g_max, FUSED, device=device) * 0.4)
    kv_deq = kv_q.float() * KV_SCALE

    total = int(s_loc.sum())
    slots = torch.randperm(total, device=device, dtype=torch.int32)
    pool = torch.zeros(total, 1, FUSED, dtype=torch.float8_e4m3fnuz, device=device)

    kv_indptr = torch.zeros(B + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(s_loc, 0)
    kv_indices = torch.empty(total, dtype=torch.int32, device=device)

    for b in range(B):
        lo = int(kv_indptr[b])
        n = int(s_loc[b])
        pos = _local_positions(int(g_len[b]), cp_world, cp_rank, device)
        sel = slots[lo : lo + n]
        pool[sel.long(), 0] = kv_q[b, pos]
        kv_indices[lo : lo + n] = sel

    T = B * q_len
    q_lat = (torch.randn(T, H, LAT, device=device) * 0.5).to(torch.bfloat16)
    q_pe = (torch.randn(T, H, ROPE, device=device) * 0.5).to(torch.bfloat16)

    return dict(
        q_lat=q_lat,
        q_pe=q_pe,
        pool=pool,
        seq_lens=s_loc,
        kv_indices=kv_indices,
        kv_indptr=kv_indptr,
        glen=g_len,
        kv_deq=kv_deq,
        T=T,
        B=B,
        q_len=q_len,
        H=H,
    )


def _reference(case, cp_world, cp_rank):
    """fp32 rank-local oracle: normalized o [T,H,512] and BASE-2 lse [T,H].

    Draft position t of request b may attend GLOBAL positions [0, G) where
    G = glen[b] - (q_len - 1 - t), restricted to those this rank holds (p % cp_world == cp_rank).
    A row with no such position keeps o = 0 and lse = -inf -- the kernel's empty-window contract.
    """
    q_lat, q_pe, kv_deq = case["q_lat"], case["q_pe"], case["kv_deq"]
    T, H, _ = q_lat.shape
    q_len = case["q_len"]

    o = torch.zeros(T, H, LAT, dtype=torch.float32, device=q_lat.device)
    lse = torch.full((T, H), float("-inf"), dtype=torch.float32, device=q_lat.device)

    for b in range(case["B"]):
        for t in range(q_len):
            end = int(case["glen"][b]) - (q_len - 1 - t)
            pos = _local_positions(end, cp_world, cp_rank, q_lat.device)
            if pos.numel() == 0:
                continue

            kv = kv_deq[b, pos]  # shape [n, 576]
            row = b * q_len + t
            score = (
                torch.einsum("hd,nd->hn", q_lat[row].float(), kv[:, :LAT])
                + torch.einsum("hd,nd->hn", q_pe[row].float(), kv[:, LAT:])
            ) * SCALE

            m = score.amax(-1)
            e = torch.exp(score - m[:, None])
            o[row] = e @ kv[:, :LAT] / e.sum(-1)[:, None]
            lse[row] = (m + torch.log(e.sum(-1))) * LOG2E

    return o, lse


def _decode(case, cp_rank, cp_world, device, with_lse=True):
    """Run the op on `case`: plain decode when cp_world is 1, else the DCP rank."""
    o_lat = torch.zeros(case["T"], case["H"], LAT, dtype=torch.bfloat16, device=device)
    lse = torch.zeros(case["T"], case["H"], dtype=torch.float32, device=device) if with_lse else None

    ma.mla_decode_a16w8(
        case["q_lat"], case["q_pe"], case["pool"], o_lat,
        case["seq_lens"], case["kv_indices"], case["kv_indptr"], SCALE, KV_SCALE,
        lse=lse, glen=case["glen"] if cp_world > 1 else None, cp_rank=cp_rank, cp_world=cp_world,
    )
    return o_lat, lse


def _check(case, o_lat, lse, cp_world, cp_rank):
    """Check against the fp32 oracle; passing `lse` None validates the output only."""
    ref_o, ref_lse = _reference(case, cp_world, cp_rank)

    assert not torch.isnan(o_lat).any()

    o_err = float((o_lat.float() - ref_o).abs().max())
    assert o_err < O_ABS, f"o_err={o_err}"

    if lse is not None:
        finite = torch.isfinite(ref_lse)
        lse_err = float(torch.where(finite, (lse - ref_lse).abs(), torch.zeros_like(lse)).max())
        assert lse_err < LSE_ABS, f"lse_err={lse_err}"
        assert torch.equal(torch.isfinite(lse), finite), "empty rank-local windows must give -inf"
    return ref_lse


# ---- plain decode and speculative verify (cp_world == 1) -------------------


@pytest.mark.gpu
@pytest.mark.parametrize("H", [1, 12, 16])
@pytest.mark.parametrize("q_len", [1, 2, 4, 8])
def test_decode_matches_reference(device, q_len, H):
    """Every draft length, at the head counts a tensor-parallel shard runs, over a ragged batch."""
    case = _build_case(3, q_len, H, 1, 0, 4096, device, seed=100 * q_len + H)
    o_lat, lse = _decode(case, 0, 1, device)
    _check(case, o_lat, lse, 1, 0)


@pytest.mark.gpu
@pytest.mark.parametrize("q_len", [1, 4])
@pytest.mark.parametrize(
    "glen",
    [[60000] + [100] * 31,
     list(range(4, 36)),
     [40000 if b in (3, 70) else 50 + b for b in range(100)]],
    ids=["one_long", "all_short", "two_waves"],
)
def test_decode_kv_split_follows_lengths(device, q_len, glen):
    """Skewed lengths exercise the length-based KV split."""
    case = _build_case(len(glen), q_len, 16, 1, 0, glen, device, seed=len(glen) + q_len)
    o_lat, lse = _decode(case, 0, 1, device)
    _check(case, o_lat, lse, 1, 0)


@pytest.mark.gpu
def test_decode_causal_window_is_end_aligned(device):
    """Position t must see exactly S - q_len + t + 1 tokens, never the whole sequence.

    Poisoning the last q_len - 1 tokens moves the last draft position's output while leaving the
    first bit-identical; under a flat causal limit, both would move.
    """
    S, q_len = 2048, 4
    case = _build_case(1, q_len, 16, 1, 0, [S], device, seed=31)
    base, _ = _decode(case, 0, 1, device)

    tail = case["kv_indices"][S - q_len + 1 :].long()  # at cp_world 1, slot i == position i
    case["pool"][tail] = _quantize_fp8(torch.randn(q_len - 1, 1, FUSED, device=device))
    o_lat, _ = _decode(case, 0, 1, device)

    assert torch.equal(o_lat[0], base[0]), "position 0 read past its causal limit"
    assert not torch.equal(o_lat[q_len - 1], base[q_len - 1]), "last position missed the tail tokens"


@pytest.mark.gpu
def test_decode_lse_is_optional(device):
    """Omitting lse leaves o_lat unchanged."""
    case = _build_case(4, 4, 16, 1, 0, 4096, device, seed=37)
    with_lse, _ = _decode(case, 0, 1, device)
    without, _ = _decode(case, 0, 1, device, with_lse=False)
    assert torch.equal(with_lse, without)


# ---- decode context parallelism (DCP) --------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize("cp_rank", [0, 3, 7])
def test_dcp_matches_reference_across_ranks(device, cp_rank):
    """Each rank of an 8-way shard reproduces its own slice of the attention."""
    case = _build_case(6, 4, 96, 8, cp_rank, 12000, device, seed=cp_rank + 1)
    o_lat, lse = _decode(case, cp_rank, 8, device)
    _check(case, o_lat, lse, 8, cp_rank)


@pytest.mark.gpu
@pytest.mark.parametrize("q_len", [1, 2, 3, 4, 5, 6, 7, 8])
def test_dcp_every_draft_length(device, q_len):
    """Every draft length shifts the per-row causal limit by a distinct amount."""
    case = _build_case(4, q_len, 96, 4, 1, 9000, device, seed=q_len)
    o_lat, lse = _decode(case, 1, 4, device)
    _check(case, o_lat, lse, 4, 1)


@pytest.mark.gpu
@pytest.mark.parametrize("H", [12, 16, 32, 70, 96, 100, 112, 128])
def test_dcp_head_domain(device, H):
    """The H domain, including sizes that do not divide the 96-row CTA slice.

    A slice is a flat run of packed (position, head) rows, so row r is (r // H, r % H) for ANY H:
    the row math never depends on the slice height.
    """
    case = _build_case(3, 5, H, 8, 5, 9000, device, seed=H)
    o_lat, lse = _decode(case, 5, 8, device)
    _check(case, o_lat, lse, 8, 5)


@pytest.mark.gpu
def test_dcp_empty_rank_local_windows(device):
    """Requests shorter than the rank's offset hold NO local KV, so o = 0 and lse = -inf."""
    B, cp_world, cp_rank = 6, 8, 7
    glen = [1, 2, 7, 8, 5000, 40000]  # the first four fall below cp_rank + 1
    case = _build_case(B, 1, 96, cp_world, cp_rank, glen, device, seed=11)
    o_lat, lse = _decode(case, cp_rank, cp_world, device)
    ref_lse = _check(case, o_lat, lse, cp_world, cp_rank)

    empty = ~torch.isfinite(ref_lse)
    assert empty.any(), "the case must actually contain an empty window"
    assert float(o_lat[empty].abs().max()) == 0.0, "an empty window must leave o_lat at zero"


@pytest.mark.gpu
def test_dcp_ragged_sequence_lengths(device):
    """A 15:1 KV-length spread in one batch: each request walks its own window."""
    B, cp_world, cp_rank = 8, 8, 2
    glen = [150000 if b % 2 else 10000 for b in range(B)]
    case = _build_case(B, 2, 96, cp_world, cp_rank, glen, device, seed=23)
    o_lat, lse = _decode(case, cp_rank, cp_world, device)
    _check(case, o_lat, lse, cp_world, cp_rank)


@pytest.mark.gpu
def test_dcp_single_request(device):
    """With B = 1 the KV split is the only source of parallelism."""
    case = _build_case(1, 1, 96, 8, 3, 60000, device, seed=29)
    o_lat, lse = _decode(case, 3, 8, device)
    _check(case, o_lat, lse, 8, 3)


# ---- supported domain / rejections -----------------------------------------


@pytest.mark.gpu
def test_rejects_head_count_above_domain(device):
    """An H beyond the domain must be rejected, not silently mis-tiled."""
    case = _build_case(2, 1, 96, 1, 0, 4000, device, seed=5)
    wide = 129
    q_lat = torch.zeros(case["T"], wide, LAT, dtype=torch.bfloat16, device=device)
    q_pe = torch.zeros(case["T"], wide, ROPE, dtype=torch.bfloat16, device=device)
    o_lat = torch.zeros(case["T"], wide, LAT, dtype=torch.bfloat16, device=device)

    with pytest.raises(ValueError, match="H must be"):
        ma.mla_decode_a16w8(q_lat, q_pe, case["pool"], o_lat, case["seq_lens"],
                            case["kv_indices"], case["kv_indptr"], SCALE, KV_SCALE)


@pytest.mark.gpu
def test_rejects_sharding_without_global_lengths(device):
    """With cp_world > 1, omitting glen leaves no causal limit to apply."""
    case = _build_case(2, 1, 16, 4, 1, 4000, device, seed=7)
    o_lat = torch.zeros(case["T"], 16, LAT, dtype=torch.bfloat16, device=device)

    with pytest.raises(ValueError, match="glen"):
        ma.mla_decode_a16w8(case["q_lat"], case["q_pe"], case["pool"], o_lat, case["seq_lens"],
                            case["kv_indices"], case["kv_indptr"], SCALE, KV_SCALE,
                            cp_rank=1, cp_world=4)


# ---- cross-rank LSE merge --------------------------------------------------


@pytest.mark.gpu
def test_cross_rank_merge_matches_reference(device):
    """Merge R ranks' (normalized partial, base-2 lse) pairs, incl. all-empty and one-empty rows."""
    R, T, HL = 8, 64, 12
    torch.manual_seed(1)

    parts_in = (torch.randn(R, T, HL, LAT, device=device) * 0.5).to(torch.bfloat16)
    lse_in = torch.randn(R, T, HL, device=device) * 4.0
    lse_in[:, : T // 8] = float("-inf")  # rows for which no rank holds any KV
    lse_in[0, T // 8 : T // 4] = float("-inf")  # rows a single rank is missing

    out = torch.empty(T, HL, LAT, dtype=torch.bfloat16, device=device)
    lse_out = torch.empty(T, HL, dtype=torch.float32, device=device)
    ma.mla_dcp_lse_merge_ranks(parts_in, lse_in, out, lse_out)

    m = lse_in.amax(0)
    live = torch.isfinite(m)
    w = torch.where(live[None], torch.exp2(lse_in - m.clamp(min=-1e30)[None]), torch.zeros_like(lse_in))
    ref = (parts_in.float() * w[..., None]).sum(0) / w.sum(0).clamp(min=1e-30)[..., None]
    ref = torch.where(live[..., None], ref, torch.zeros_like(ref))
    ref_lse = torch.log2(w.sum(0)) + m
    ref_lse[~live] = float("-inf")

    assert (out.float() - ref).abs().max() < O_ABS
    assert (lse_out - ref_lse).abs()[live].max() < MERGE_ABS
    assert torch.equal(torch.isfinite(lse_out), live), "empty rows must merge to -inf"


@pytest.mark.gpu
@pytest.mark.parametrize("R", [1, 2, 3, 5, 8, 16])
def test_cross_rank_merge_rank_counts(device, R):
    """Every rank count the launcher templates over, non-powers of two included."""
    T, HL = 16, 8
    torch.manual_seed(R)

    parts_in = (torch.randn(R, T, HL, LAT, device=device) * 0.5).to(torch.bfloat16)
    lse_in = torch.randn(R, T, HL, device=device) * 3.0
    out = torch.empty(T, HL, LAT, dtype=torch.bfloat16, device=device)
    ma.mla_dcp_lse_merge_ranks(parts_in, lse_in, out)

    w = torch.exp2(lse_in - lse_in.amax(0)[None])
    ref = (parts_in.float() * w[..., None]).sum(0) / w.sum(0)[..., None]
    assert (out.float() - ref).abs().max() < O_ABS


# ── CUDA graph capture ────────────────────────────────────────────────────────────────────────
#
# The host chooses the row tile, the KV part count, the merge split and the non-temporal hint, all
# from tensor shapes the caller already knows -- never from a host read of a device tensor. That is
# what makes the op capturable: the launch configuration is fixed as soon as the shapes are, and
# nothing on the launch path synchronizes.
#
# The trade-off is the contract these tests pin down: a captured graph is bound to its SHAPE, not
# to its CONTENT. Sequence lengths, page tables and global lengths all live in device memory and are
# read by the kernel, so a replay must observe new values without re-capturing.


def _capture(fn):
    """Warm up on a side stream (the allocator requires it), then capture `fn` into a graph."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g


@pytest.mark.gpu
@pytest.mark.parametrize(
    "B,q_len,H,cp_world",
    [(16, 1, 96, 8),    # NT=6, one slice, 19 KV parts
     (16, 4, 96, 8),    # NT=6, four slices
     (16, 1, 128, 8),   # NT=4 -- another decode instantiation
     (32, 1, 16, 8),    # NT=1, with the merge splitting its dims
     (4, 1, 12, 8),     # tiny: partial slice, exercising the AllTilesLive=false arm
     (32, 1, 16, 1),    # plain decode, no lse
     (8, 4, 16, 1)],    # speculative verify, no lse
)
def test_cuda_graph_replay_matches_eager(device, B, q_len, H, cp_world):
    """Every tile / part / merge-split the host can choose must survive capture and replay."""
    cp_rank = 3 if cp_world > 1 else 0
    case = _build_case(B, q_len, H, cp_world, cp_rank, 40000, device, seed=17)
    o_lat = torch.zeros(case["T"], H, LAT, dtype=torch.bfloat16, device=device)
    lse = torch.zeros(case["T"], H, dtype=torch.float32, device=device) if cp_world > 1 else None

    def run():
        ma.mla_decode_a16w8(
            case["q_lat"], case["q_pe"], case["pool"], o_lat,
            case["seq_lens"], case["kv_indices"], case["kv_indptr"], SCALE, KV_SCALE,
            lse=lse, glen=case["glen"] if cp_world > 1 else None, cp_rank=cp_rank, cp_world=cp_world)

    g = _capture(run)
    o_lat.zero_()
    if lse is not None:
        lse.zero_()
    g.replay()
    torch.cuda.synchronize()

    _check(case, o_lat, lse, cp_world, cp_rank)


@pytest.mark.gpu
def test_cuda_graph_replay_follows_device_lengths(device):
    """A replay must read the CURRENT sequence lengths, not those present at capture time."""
    B, q_len, H, cp_world, cp_rank = 8, 2, 96, 8, 3
    long_case = _build_case(B, q_len, H, cp_world, cp_rank, 40000, device, seed=23)
    o_lat = torch.zeros(long_case["T"], H, LAT, dtype=torch.bfloat16, device=device)
    lse = torch.zeros(long_case["T"], H, dtype=torch.float32, device=device)

    def run():
        ma.mla_decode_a16w8(
            long_case["q_lat"], long_case["q_pe"], long_case["pool"], o_lat,
            long_case["seq_lens"], long_case["kv_indices"], long_case["kv_indptr"], SCALE, KV_SCALE,
            lse=lse, glen=long_case["glen"], cp_rank=cp_rank, cp_world=cp_world)

    g = _capture(run)

    # Same shapes, shorter sequences: build a fresh case and copy its device tensors into the
    # captured buffers. Only the CONTENT changes, so the graph remains valid.
    short_case = _build_case(B, q_len, H, cp_world, cp_rank, 9000, device, seed=29)
    for key in ("q_lat", "q_pe", "seq_lens", "kv_indptr", "glen"):
        long_case[key].copy_(short_case[key])
    # The short case indexes a smaller pool, so its rows and page table land in the prefix of the
    # captured (larger) buffers and every index it uses remains in range.
    n_rows = short_case["pool"].shape[0]
    long_case["pool"][:n_rows].copy_(short_case["pool"])
    long_case["kv_indices"][: short_case["kv_indices"].numel()].copy_(short_case["kv_indices"])

    o_lat.zero_(); lse.zero_()
    g.replay()
    torch.cuda.synchronize()

    _check(short_case, o_lat, lse, cp_world, cp_rank)
