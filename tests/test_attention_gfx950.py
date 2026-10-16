"""gfx950 (CDNA4) attention: ma.forward (BSHD / BHSD batch) and ma.forward_varlen (THD) against an fp64 reference.

Head dims (128, 128) and (192, 128) (MLA prefill), dense and bottom-right causal, GQA, lengths off the
kernel's 48/64/384 grids, Sq != Skv, zero-length sequences, and grids small enough to split KV. Equal
lengths via forward_varlen must match forward bit for bit, and every strided view (BHSD, slices,
padding, packed QKV, per-tensor layouts, head-major varlen) must match its contiguous copy.
return_lse's LSE (AITER's layouts) against fp64; GPU cu_seqlens (no host sync, graph replays with new
lengths) and VarlenPlan (plan once, run per layer) against the host-planned path.

Skipped unless the current GPU is gfx950 and moonmath_amd._C_gfx950 is built.
"""

import inspect
import itertools
import random
import types

import pytest
import torch

import moonmath_amd as ma
from moonmath_amd import _arch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not _arch.is_built("gfx950")
    or _arch.get_gfx_runtime() != "gfx950",
    reason="needs a gfx950 GPU and moonmath_amd._C_gfx950",
)

HEAD_DIMS = {"d128": (128, 128), "mla": (192, 128)}
WORST, RMS = (
    2e-2,
    6e-3,
)  # max |err| / max |ref| and rms(err) / rms(ref), matching the kernel's own checkers


def _randn(gen, *shape):
    return torch.randn(*shape, device="cuda", generator=gen).bfloat16()


def _batch(kind, B, Sq, Skv, H, HKv, seed=0):
    dqk, dv = HEAD_DIMS[kind]
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (
        _randn(g, B, Sq, H, dqk),
        _randn(g, B, Skv, HKv, dqk),
        _randn(g, B, Skv, HKv, dv),
    )


def _packed(kind, sq, skv, H, HKv, seed=0):
    dqk, dv = HEAD_DIMS[kind]
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (
        _randn(g, sum(sq), H, dqk),
        _randn(g, sum(skv), HKv, dqk),
        _randn(g, sum(skv), HKv, dv),
    )


def _cu(lens):
    return torch.tensor([0, *itertools.accumulate(lens)], dtype=torch.int32)


def _reference(q, k, v, causal, scale):
    """fp64 attention for one sequence: q [Sq, H, Dqk], k [Skv, HKv, Dqk], v [Skv, HKv, Dv]. Rows with no
    visible key are 0."""
    Sq, H, Skv, HKv = q.shape[0], q.shape[1], k.shape[0], k.shape[1]
    out = torch.zeros(Sq, H, v.shape[-1], dtype=torch.float64, device="cuda")
    if Sq == 0 or Skv == 0:
        return out
    visible = None
    if causal:
        rows = torch.arange(Sq, device="cuda")[:, None] + (Skv - Sq)
        visible = torch.arange(Skv, device="cuda")[None, :] <= rows
    for h in range(H):
        kh = h // (H // HKv)
        s = q[:, h].double() @ k[:, kh].double().T * scale
        if visible is not None:
            s = s.masked_fill(~visible, float("-inf"))
        p = torch.softmax(s, -1).nan_to_num(
            0.0
        )  # row with no key: all -inf -> NaN -> 0
        out[:, h] = p @ v[:, kh].double()
    return out


def _assert_close(got, ref):
    if got.numel() == 0:
        return
    assert torch.isfinite(got).all(), "non-finite output"
    err = (got.double() - ref).abs()
    worst = float(err.max() / ref.abs().max().clamp(min=1e-6))
    rms = float(err.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt().clamp(min=1e-6))
    assert worst < WORST and rms < RMS, f"worst {worst:.2e} rms {rms:.2e}"
    empty = (ref == 0).all(-1)
    assert (got[empty] == 0).all(), "rows that see no key must be 0"


def _bits(t):
    return t.view(torch.int16)


def _fwd(q, k, v, **kw):
    """ma.forward on BSHD tensors (the default layout being BHSD)."""
    return ma.forward(q, k, v, layout="bshd", **kw)


# ---- ma.forward: a BSHD batch ---------------------------------------------------------------------

BATCH_SHAPES = [  # fields: B, Sq, Skv, H, HKv
    (1, 1, 1, 1, 1),
    (2, 47, 47, 8, 2),  # GQA group of 4
    (1, 385, 1000, 4, 4),  # Sq < Skv: prefix case
    (2, 1000, 385, 6, 3),  # Sq > Skv: causal rows with no visible key
    (3, 777, 777, 12, 4),
    (1, 3000, 9000, 2, 2),  # few CTAs: exercises KV split
    (1, 8192, 8192, 2, 1),  # small dense grid: exercises KV split
]


@pytest.mark.parametrize("kind", HEAD_DIMS)
@pytest.mark.parametrize("causal", [False, True], ids=["dense", "causal"])
@pytest.mark.parametrize(
    "shape", BATCH_SHAPES, ids=lambda s: "B{}_Sq{}_Skv{}_H{}_{}".format(*s)
)
def test_forward_matches_reference(kind, causal, shape):
    B, Sq, Skv, H, HKv = shape
    q, k, v = _batch(kind, *shape)
    scale = q.shape[-1] ** -0.5
    out = _fwd(q, k, v, causal=causal)
    assert out.shape == (B, Sq, H, v.shape[-1]) and out.dtype == torch.bfloat16
    for b in range(B):
        _assert_close(out[b], _reference(q[b], k[b], v[b], causal, scale))


def _stress(data, kind, B, Sq, Skv, H, HKv):
    """Inputs that shift the softmax max (bench/accuracy.py's data kinds): peaked rows, a max rising along
    the keys (forcing rescales), and an attention sink above a far lower bulk."""
    dqk, dv = HEAD_DIMS[kind]
    g = torch.Generator(device="cuda").manual_seed(3)
    q = torch.randn(B, Sq, H, dqk, device="cuda", generator=g)
    k = torch.randn(B, Skv, HKv, dqk, device="cuda", generator=g)
    v = torch.randn(B, Skv, HKv, dv, device="cuda", generator=g)
    u = torch.randn(dqk, device="cuda", generator=g)
    u /= u.norm()
    if data == "qk_x8":
        q, k = q * 8, k * 8
    elif data == "ramp":
        q = q + 3.0 * u
        k = k + torch.linspace(0, 1, Skv, device="cuda").view(1, Skv, 1, 1) * 40.0 * u
    elif data == "sink_tail":
        q, k = 0.5 * q + 2.0 * u, k - 20.0 * u
        k[:, 0] = torch.randn(B, HKv, dqk, device="cuda", generator=g) + 45.0 * u
        v = v + 2.0 * torch.randn(1, 1, HKv, dv, device="cuda", generator=g)
    return q.bfloat16(), k.bfloat16(), v.bfloat16()


@pytest.mark.parametrize("kind", HEAD_DIMS)
@pytest.mark.parametrize("data", ["qk_x8", "ramp", "sink_tail"])
def test_forward_stress_data(kind, data):
    q, k, v = _stress(data, kind, 2, 1000, 3000, 8, 2)
    out = _fwd(q, k, v, causal=True)
    for b in range(2):
        _assert_close(out[b], _reference(q[b], k[b], v[b], True, q.shape[-1] ** -0.5))


@pytest.mark.parametrize("kind", HEAD_DIMS)
def test_forward_softmax_scale(kind):
    q, k, v = _batch(kind, 2, 300, 500, 4, 2)
    out = _fwd(q, k, v, causal=True, softmax_scale=0.3)
    for b in range(2):
        _assert_close(out[b], _reference(q[b], k[b], v[b], True, 0.3))


@pytest.mark.parametrize("kind", HEAD_DIMS)
def test_forward_empty_extents(kind):
    q, k, v = _batch(kind, 2, 5, 0, 4, 4)
    assert (_fwd(q, k, v) == 0).all(), "no keys: every row is 0"
    q, k, v = _batch(kind, 2, 0, 7, 4, 4)
    assert _fwd(q, k, v).shape == (2, 0, 4, 128)
    q, k, v = _batch(kind, 0, 5, 7, 4, 4)
    assert _fwd(q, k, v).shape == (0, 5, 4, 128)


# ---- ma.forward_varlen: THD packing ---------------------------------------------------------------

VARLEN_CASES = [  # fields: sq, skv, H, HKv, causal
    ([0, 5, 0, 300], [0, 5, 7, 300], 4, 4, True),
    ([0, 5, 0, 300], [0, 5, 7, 300], 4, 4, False),
    ([100, 0, 1], [0, 64, 1], 2, 1, False),  # skv 0: zero-length rows
    ([385, 383, 47, 49, 1000], [385, 1000, 47, 4000, 1000], 8, 2, True),
    ([1777], [1000], 4, 4, True),  # Sq > Skv: rows seeing no key
    # a prefix chunk plus prompts
    ([4096, 512, 512, 777], [12288, 512, 600, 777], 12, 12, True),
    ([3000], [9000], 2, 2, True),  # few CTAs: exercises KV split
    ([384 * 3 + 1] * 3, [5000, 70, 384 * 3 + 1], 16, 2, True),
]


def _random_cases(n):
    rng = random.Random(1)
    cases = []
    for _ in range(n):
        H, HKv = rng.choice([(1, 1), (4, 4), (8, 2), (16, 4), (6, 3)])
        causal = rng.random() < 0.7
        sq, skv = [], []
        for _ in range(rng.choice([1, 2, 5, 17])):
            s = rng.choice(
                [0, 1, rng.randint(1, 64), rng.randint(1, 1000), rng.randint(300, 3000)]
            )
            sq.append(s)
            skv.append(
                s + rng.choice([0, 0, rng.randint(0, 2000)])
                if causal
                else rng.choice([s, rng.randint(0, 3000)])
            )
        cases.append((sq, skv, H, HKv, causal))
    return cases


def _check_varlen(q, k, v, sq, skv, causal, out):
    cq, ck = _cu(sq), _cu(skv)
    scale = q.shape[-1] ** -0.5
    for i in range(len(sq)):
        qs, ks = slice(cq[i], cq[i + 1]), slice(ck[i], ck[i + 1])
        _assert_close(out[qs], _reference(q[qs], k[ks], v[ks], causal, scale))


@pytest.mark.parametrize("kind", HEAD_DIMS)
@pytest.mark.parametrize(
    "case",
    VARLEN_CASES + _random_cases(6),
    ids=lambda c: f"n{len(c[0])}_H{c[2]}_{c[3]}_{'causal' if c[4] else 'dense'}",
)
@pytest.mark.parametrize("cu_on", ["cpu", "cuda"])
def test_forward_varlen_matches_reference(kind, case, cu_on):
    sq, skv, H, HKv, causal = case
    q, k, v = _packed(kind, sq, skv, H, HKv)
    cu_q, cu_k = _cu(sq).to(cu_on), _cu(skv).to(cu_on)
    out = ma.forward_varlen(q, k, v, cu_q, cu_k, max(sq), max(skv), causal=causal)
    assert out.shape == (sum(sq), H, v.shape[-1])
    _check_varlen(q, k, v, sq, skv, causal, out)


@pytest.mark.parametrize("kind", HEAD_DIMS)
@pytest.mark.parametrize(
    "shape",
    [
        (3, 1000, 1777, 8, 2, False),
        (2, 4096, 4096, 16, 16, True),
        (1, 8192, 8192, 8, 8, False),
        (1, 3000, 5000, 4, 4, True),
        (4, 385, 385, 12, 4, True),
    ],
    ids=str,
)
def test_varlen_equals_batched_bitwise(kind, shape):
    B, Sq, Skv, H, HKv, causal = shape
    q, k, v = _batch(kind, B, Sq, Skv, H, HKv, seed=7)
    batched = _fwd(q, k, v, causal=causal)
    varlen = ma.forward_varlen(
        q.flatten(0, 1),
        k.flatten(0, 1),
        v.flatten(0, 1),
        _cu([Sq] * B),
        _cu([Skv] * B),
        Sq,
        Skv,
        causal=causal,
    )
    assert torch.equal(_bits(varlen), _bits(batched.flatten(0, 1)))


# ---- return_lse: fp64 reference, AITER's layouts --------------------------------------------------

LSE_ATOL = 5e-3  # kernel sums the bf16 P it multiplies V by: within ln(1 + 2^-8) of the exact LSE


def _reference_lse(q, k, causal, scale):
    """fp64 log-sum-exp of one sequence's scaled scores, [H, Sq]; -inf when a row has no visible key."""
    Sq, H, Skv, HKv = q.shape[0], q.shape[1], k.shape[0], k.shape[1]
    lse = torch.empty(H, Sq, dtype=torch.float64, device="cuda")
    for h in range(H):
        s = q[:, h].double() @ k[:, h // (H // HKv)].double().T * scale
        if causal:
            rows = torch.arange(Sq, device="cuda")[:, None] + (Skv - Sq)
            s = s.masked_fill(
                torch.arange(Skv, device="cuda")[None, :] > rows, float("-inf")
            )
        lse[h] = torch.logsumexp(s, -1)
    return lse


def _assert_lse_close(got, ref):
    assert got.dtype == torch.float32 and got.shape == ref.shape
    empty = ref == float("-inf")
    assert (got[empty] == float("-inf")).all(), "a row that sees no key has lse -inf"
    if (~empty).any():
        err = float((got[~empty].double() - ref[~empty]).abs().max())
        assert err < LSE_ATOL, f"lse error {err:.2e}"


@pytest.mark.parametrize("kind", HEAD_DIMS)
@pytest.mark.parametrize("causal", [False, True], ids=["dense", "causal"])
@pytest.mark.parametrize(
    "shape", BATCH_SHAPES, ids=lambda s: "B{}_Sq{}_Skv{}_H{}_{}".format(*s)
)
def test_forward_lse(kind, causal, shape):
    B, Sq, Skv, H, HKv = shape
    q, k, v = _batch(kind, *shape)
    out, lse = _fwd(q, k, v, causal=causal, return_lse=True)
    assert lse.shape == (B, H, Sq) and lse.is_contiguous()  # AITER's [B, H, Sq] layout
    assert torch.equal(_bits(out), _bits(_fwd(q, k, v, causal=causal)))
    for b in range(B):
        _assert_lse_close(
            lse[b], _reference_lse(q[b], k[b], causal, q.shape[-1] ** -0.5)
        )
    # BHSD tensors give the same [B, H, Sq] lse
    _, lse_bhsd = ma.forward(*_bhsd(q, k, v), causal=causal, return_lse=True)
    assert torch.equal(lse_bhsd, lse)


def test_forward_lse_no_keys():
    q, k, v = _batch("d128", 2, 5, 0, 4, 4)
    out, lse = _fwd(q, k, v, return_lse=True)
    assert (out == 0).all() and (lse == float("-inf")).all()


def _check_varlen_lse(q, k, sq, skv, causal, lse):
    cq, ck = _cu(sq), _cu(skv)
    assert lse.shape == (q.shape[1], q.shape[0])  # AITER's [H, total_q] layout
    for i in range(len(sq)):
        qs, ks = slice(cq[i], cq[i + 1]), slice(ck[i], ck[i + 1])
        _assert_lse_close(
            lse[:, qs], _reference_lse(q[qs], k[ks], causal, q.shape[-1] ** -0.5)
        )


@pytest.mark.parametrize("kind", HEAD_DIMS)
@pytest.mark.parametrize(
    "case",
    VARLEN_CASES[:5] + VARLEN_CASES[6:],
    ids=lambda c: f"n{len(c[0])}_H{c[2]}_{c[3]}_{'causal' if c[4] else 'dense'}",
)
@pytest.mark.parametrize("path", ["host cu", "device cu", "plan"])
def test_forward_varlen_lse(kind, case, path):
    sq, skv, H, HKv, causal = case
    q, k, v = _packed(kind, sq, skv, H, HKv)
    args = (_cu(sq), _cu(skv), max(sq), max(skv))
    if path == "host cu":
        out, lse = ma.forward_varlen(q, k, v, *args, causal=causal, return_lse=True)
    else:
        args = (args[0].cuda(), args[1].cuda(), *args[2:])
        if path == "device cu":
            out, lse = ma.forward_varlen(q, k, v, *args, causal=causal, return_lse=True)
        else:
            plan = ma.VarlenPlan().plan(
                *args, q.shape[0], k.shape[0], H, HKv, HEAD_DIMS[kind], causal
            )
            out, lse = plan.run(q, k, v, return_lse=True)
    _check_varlen(q, k, v, sq, skv, causal, out)
    _check_varlen_lse(q, k, sq, skv, causal, lse)


# ---- GPU cu_seqlens and VarlenPlan: no host sync, graph replay, == host-planned path ---------------

SEQS = ([385, 0, 47, 1000], [1000, 64, 47, 1000])  # (sq, skv)


def _device_args(sq, skv):
    """forward_varlen's sequence arguments, with cu_seqlens placed on the GPU."""
    return _cu(sq).cuda(), _cu(skv).cuda(), max(sq), max(skv)


def _host_copies(sq, skv):
    return dict(host_cu_seqlens_q=_cu(sq), host_cu_seqlens_k=_cu(skv))


def _plan(kind, q, k, sq, skv, causal=True, **host):
    return ma.VarlenPlan().plan(
        *_device_args(sq, skv),
        q.shape[0],
        k.shape[0],
        q.shape[1],
        k.shape[1],
        HEAD_DIMS[kind],
        causal,
        **host,
    )


@pytest.mark.filterwarnings("ignore:Synchronization debug mode")
def test_device_cu_seqlens_never_sync():
    """torch's sync debug mode catches torch-level syncs (e.g. .cpu() on cu_seqlens); the graph tests below
    prove the kernels' own host code doesn't sync either (a capture would fail)."""
    sq, skv = SEQS
    q, k, v = _packed("d128", sq, skv, 8, 2)
    args = _device_args(sq, skv)
    plan = ma.VarlenPlan()
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        ma.forward_varlen(q, k, v, *args, causal=True, return_lse=True)
        plan.plan(*args, q.shape[0], k.shape[0], 8, 2, (128, 128), True)
        plan.run(q, k, v, return_lse=True)
    finally:
        torch.cuda.set_sync_debug_mode("default")


@pytest.mark.parametrize("kind", HEAD_DIMS)
def test_host_copies_equal_host_cu_bitwise(kind):
    """GPU cu_seqlens plus host copies plan on the host exactly like host cu_seqlens: identical bits."""
    sq, skv = SEQS
    q, k, v = _packed(kind, sq, skv, 8, 2)
    want, want_lse = ma.forward_varlen(
        q, k, v, _cu(sq), _cu(skv), max(sq), max(skv), causal=True, return_lse=True
    )
    dev, host = _device_args(sq, skv), _host_copies(sq, skv)
    got, lse = ma.forward_varlen(q, k, v, *dev, causal=True, return_lse=True, **host)
    assert torch.equal(_bits(got), _bits(want)) and torch.equal(lse, want_lse)
    got, lse = _plan(kind, q, k, sq, skv, **host).run(q, k, v, return_lse=True)
    assert torch.equal(_bits(got), _bits(want)) and torch.equal(lse, want_lse)


@pytest.mark.parametrize("kind", HEAD_DIMS)
@pytest.mark.parametrize("host_copies", [False, True], ids=["device plan", "host plan"])
def test_plan_run_equals_one_shot_bitwise(kind, host_copies):
    """A single plan over three layers' tensors: each run matches forward_varlen with the same arguments."""
    sq, skv = SEQS
    host = _host_copies(sq, skv) if host_copies else {}
    q, k, _ = _packed(kind, sq, skv, 8, 2)
    plan = _plan(kind, q, k, sq, skv, **host)
    for layer in range(3):
        q, k, v = _packed(kind, sq, skv, 8, 2, seed=layer)
        want, want_lse = ma.forward_varlen(
            q, k, v, *_device_args(sq, skv), causal=True, return_lse=True, **host
        )
        got, lse = plan.run(q, k, v, return_lse=True)
        assert torch.equal(_bits(got), _bits(want)) and torch.equal(lse, want_lse)
    plan.close()


@pytest.mark.parametrize("kind", HEAD_DIMS)
@pytest.mark.parametrize("how", ["plan + runs", "one-shot"])
def test_graph_replays_new_cu_seqlens(kind, how):
    """A graph captured once can replay with new lengths: cu_seqlens are read on the GPU each replay.
    Totals and max_seqlen are fixed; host copies are ignored during capture."""
    Tq, Tk, max_q, max_k, H, HKv = 1500, 3000, 1500, 3000, 8, 2
    lengths = [  # (sq, skv) with sums == totals
        ([500, 500, 500], [1000, 1000, 1000]),
        ([1000, 0, 500], [1500, 500, 1000]),
        ([1, 1, 1498], [1, 1, 2998]),
        ([0, 1500, 0], [0, 3000, 0]),
    ]
    layers = [_packed(kind, [Tq], [Tk], H, HKv, seed=i) for i in range(3)]
    cu_q, cu_k = _cu(lengths[0][0]).cuda(), _cu(lengths[0][1]).cuda()
    dev = (cu_q, cu_k, max_q, max_k)
    stale = _host_copies(*lengths[0])

    graph = torch.cuda.CUDAGraph()
    plan = ma.VarlenPlan()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        if how == "plan + runs":
            plan.plan(*dev, Tq, Tk, H, HKv, HEAD_DIMS[kind], True, **stale)
            outs = [plan.run(q, k, v, return_lse=True) for q, k, v in layers]
        else:
            outs = [
                ma.forward_varlen(q, k, v, *dev, causal=True, return_lse=True, **stale)
                for q, k, v in layers
            ]
    for sq, skv in lengths[1:]:
        cu_q.copy_(_cu(sq)), cu_k.copy_(_cu(skv))
        graph.replay()
        new = (_cu(sq).cuda(), _cu(skv).cuda(), max_q, max_k)
        for (q, k, v), (out, lse) in zip(layers, outs):
            want, want_lse = ma.forward_varlen(
                q, k, v, *new, causal=True, return_lse=True
            )
            assert torch.equal(_bits(out), _bits(want)) and torch.equal(lse, want_lse)
        q, k, v = layers[0]
        _check_varlen(q, k, v, sq, skv, True, outs[0][0])
        _check_varlen_lse(q, k, sq, skv, True, outs[0][1])
    del graph
    plan.close()


@pytest.mark.filterwarnings("ignore:The CUDA Graph is empty")
def test_plan_and_run_share_a_capture():
    sq, skv = SEQS
    q, k, v = _packed("d128", sq, skv, 8, 2)
    args = (*_device_args(sq, skv), q.shape[0], k.shape[0], 8, 2, (128, 128), True)
    plan = ma.VarlenPlan().plan(*args)  # eager planning
    torch.cuda.synchronize()
    with pytest.raises(ValueError, match="same graph capture"):
        with torch.cuda.graph(torch.cuda.CUDAGraph()):
            plan.run(q, k, v)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        plan.plan(*args)
    with pytest.raises(ValueError, match="same graph capture"):
        plan.run(q, k, v)  # eager, on a plan created inside the capture
    plan.plan(*args)
    plan.run(q, k, v)  # re-planned eagerly: now fine
    del graph


def test_plan_closed_during_a_capture():
    """close() (or garbage collection) in another graph's capture waits: freeing then would break it."""
    sq, skv = SEQS
    q, k, v = _packed("d128", sq, skv, 8, 2)
    plan = _plan("d128", q, k, sq, skv)
    plan.run(q, k, v)
    want = _fwd(*_batch("d128", 1, 300, 400, 4, 2), causal=True)
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    inputs = _batch("d128", 1, 300, 400, 4, 2)
    with torch.cuda.graph(graph):
        out = _fwd(*inputs, causal=True)
        plan.close()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(_bits(out), _bits(want))
    with pytest.raises(RuntimeError, match="plan"):
        plan.run(q, k, v)
    plan.plan(
        *_device_args(sq, skv), q.shape[0], k.shape[0], 8, 2, (128, 128), True
    )  # fresh plan
    plan.run(q, k, v)
    plan.close()  # also frees the one closed inside the capture


# ---- ma.forward_paged: paged K/V, bit-for-bit equal to forward_varlen on the same K/V --------------

PAGE_SIZES = {"d128": [8, 16, 64], "mla": [16, 64]}
PAGED_CASES = [  # fields: sq, skv (prefix + new tokens), H, HKv, causal
    ([64], [64], 4, 4, True),
    ([100, 0, 37], [300, 5, 37], 8, 2, True),  # a sequence with no q rows
    ([1000, 1], [1777, 1000], 4, 1, True),
    ([500, 300], [500, 900], 6, 3, False),
    ([3000], [9000], 2, 2, True),  # few CTAs: exercises KV split
]


def _paged(kind, page, sq, skv, H, HKv, seed=0):
    """Packed q, k, v; NaN-filled caches [pages, page, HKv, D] with k and v in shuffled pages plus spare
    pages; each sequence's pages given as kv_indptr and kv_page_indices."""
    q, k, v = _packed(kind, sq, skv, H, HKv, seed)
    counts = [-(-s // page) for s in skv]
    indptr, pool = _cu(counts), sum(counts) + 3
    pages = torch.randperm(pool, generator=torch.Generator().manual_seed(seed))
    pages = pages[: sum(counts)]
    nan = float("nan")
    caches = [
        torch.full((pool, page, *t.shape[1:]), nan, device="cuda").bfloat16()
        for t in (k, v)
    ]
    ck = _cu(skv)
    for i, s in enumerate(skv):
        r = torch.arange(s)
        slots = pages[indptr[i] + r // page] * page + r % page
        for cache, t in zip(caches, (k, v)):
            cache.view(-1, *t.shape[1:])[slots.cuda()] = t[ck[i] : ck[i + 1]]
    return q, k, v, *caches, indptr, pages.int().cuda()


@pytest.mark.parametrize(
    "kind, page",
    [(kind, p) for kind, pages in PAGE_SIZES.items() for p in pages],
    ids=str,
)
@pytest.mark.parametrize(
    "case",
    PAGED_CASES,
    ids=lambda c: f"n{len(c[0])}_H{c[2]}_{c[3]}_{'causal' if c[4] else 'dense'}",
)
def test_forward_paged_equals_varlen(kind, page, case):
    sq, skv, H, HKv, causal = case
    q, k, v, k_cache, v_cache, indptr, table = _paged(kind, page, sq, skv, H, HKv)
    out, lse = ma.forward_paged(
        q,
        k_cache,
        v_cache,
        _cu(sq),
        _i32(*skv),
        kv_indptr=indptr,
        kv_page_indices=table,
        causal=causal,
        return_lse=True,
    )
    ref, ref_lse = ma.forward_varlen(
        q, k, v, _cu(sq), _cu(skv), max(sq), max(skv), causal=causal, return_lse=True
    )
    assert torch.equal(_bits(out), _bits(ref)) and torch.equal(lse, ref_lse)
    _check_varlen(q, k, v, sq, skv, causal, out)


@pytest.mark.parametrize("kind", HEAD_DIMS)
def test_forward_paged_block_table_and_gpu_lengths(kind):
    """vLLM's block_table (D128: K and V as views of one [pages, 2, page, HKv, D] cache) and GPU
    lengths give the kv_indptr call's bits."""
    sq, skv, H, HKv, page = [300, 0, 64], [1000, 7, 64], 8, 2, 16
    q, k, v, k_cache, v_cache, indptr, table = _paged(kind, page, sq, skv, H, HKv)
    want = ma.forward_paged(
        q,
        k_cache,
        v_cache,
        _cu(sq),
        _i32(*skv),
        kv_indptr=indptr,
        kv_page_indices=table,
        causal=True,
    )
    rows = max(indptr.diff())
    block_table = torch.zeros(len(sq), rows, dtype=torch.int32, device="cuda")
    for i in range(len(sq)):
        block_table[i, : indptr[i + 1] - indptr[i]] = table[indptr[i] : indptr[i + 1]]
    if kind == "d128":
        kv = torch.stack([k_cache, v_cache], 1)
        k_cache, v_cache = kv[:, 0], kv[:, 1]
    got = ma.forward_paged(
        q,
        k_cache,
        v_cache,
        _cu(sq).cuda(),
        _i32(*skv).cuda(),
        block_table=block_table,
        causal=True,
    )
    assert torch.equal(_bits(got), _bits(want))


# ---- layouts and strided views: any view must equal its contiguous copy bit for bit ----------------


def _bhsd(*ts):
    return [t.transpose(1, 2).contiguous() for t in ts]


@pytest.mark.parametrize("kind", HEAD_DIMS)
@pytest.mark.parametrize("causal", [False, True], ids=["dense", "causal"])
@pytest.mark.parametrize(
    "shape",
    [
        (2, 47, 47, 8, 2),
        (2, 1000, 385, 6, 3),
        (1, 3000, 9000, 2, 2),
        (3, 777, 777, 12, 4),
    ],
    ids=lambda s: "B{}_Sq{}_Skv{}_H{}_{}".format(*s),
)
def test_bhsd_equals_bshd_bitwise(kind, causal, shape):
    q, k, v = _batch(kind, *shape, seed=11)
    want = _fwd(q, k, v, causal=causal)
    qh, kh, vh = _bhsd(q, k, v)
    got = ma.forward(qh, kh, vh, causal=causal)  # BHSD is the default
    B, Sq, Skv, H, HKv = shape
    assert got.shape == (B, H, Sq, 128) and got.is_contiguous()
    assert torch.equal(_bits(got), _bits(want.transpose(1, 2)))
    # the transposed views, read in either direction
    views = ma.forward(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal=causal
    )
    assert torch.equal(_bits(views), _bits(got))
    views = _fwd(
        qh.transpose(1, 2), kh.transpose(1, 2), vh.transpose(1, 2), causal=causal
    )
    assert torch.equal(_bits(views), _bits(want))


def test_bhsd_matches_reference():
    q, k, v = _batch("d128", 2, 385, 1000, 8, 2)
    out = ma.forward(*_bhsd(q, k, v), causal=True)
    for b in range(2):
        _assert_close(
            out[b].transpose(0, 1), _reference(q[b], k[b], v[b], True, 128**-0.5)
        )


def _views(kind, B, Sq, Skv, H, HKv, how):
    """BSHD views of one data set, laid out in memory by `how`."""
    dqk, dv = HEAD_DIMS[kind]
    g = torch.Generator(device="cuda").manual_seed(5)
    if how == "padded":  # head dims in wider rows, tokens in a longer buffer
        q = _randn(g, B, Sq + 9, H, dqk + 64)[:, 3 : 3 + Sq, :, :dqk]
        k = _randn(g, B, Skv + 5, HKv, dqk + 8)[:, 5:, :, 8:]
        v = _randn(g, B, Skv, HKv + 3, dv + 16)[:, :, 1 : 1 + HKv, 16:]
    elif how == "head slices":  # a subset of a wider head set
        q = _randn(g, B, Sq, H + 4, dqk)[:, :, 2 : 2 + H]
        k = _randn(g, B, Skv, 2 * HKv, dqk)[:, :, ::2]
        v = _randn(g, B, Skv, 2 * HKv, dv)[:, :, 1::2]
    elif how == "mixed":  # every tensor with a distinct memory order
        q = _randn(g, B, H, Sq, dqk).transpose(1, 2)  # BHSD storage
        k = _randn(g, B, Skv, HKv, dqk)  # BSHD storage
        v = _randn(g, Skv, B, HKv, dv).transpose(0, 1)  # SBHD storage
    elif how == "packed kv":  # MLA-style: k and v rows in one buffer
        kv = _randn(g, B, Skv, HKv, dqk + dv)
        q, k, v = _randn(g, B, Sq, H, dqk), kv[..., :dqk], kv[..., dqk:]
    elif how == "unaligned":  # shifted by 2 bytes

        def off(*shape):
            n = 1
            for d in shape:
                n *= d
            return _randn(g, n + 1)[1:].view(*shape)

        q, k, v = off(B, Sq, H, dqk), off(B, Skv, HKv, dqk), off(B, Skv, HKv, dv)
    return q, k, v


@pytest.mark.parametrize("kind", HEAD_DIMS)
@pytest.mark.parametrize(
    "how", ["padded", "head slices", "mixed", "packed kv", "unaligned"]
)
@pytest.mark.parametrize("causal", [False, True], ids=["dense", "causal"])
def test_strided_views_equal_contiguous_bitwise(kind, how, causal):
    q, k, v = _views(kind, 2, 1000, 1777, 8, 2, how)
    assert how == "unaligned" or not (
        q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    )
    want = _fwd(q.contiguous(), k.contiguous(), v.contiguous(), causal=causal)
    assert torch.equal(_bits(_fwd(q, k, v, causal=causal)), _bits(want))
    # same views, as BHSD
    got = ma.forward(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal=causal
    )
    assert torch.equal(_bits(got), _bits(want.transpose(1, 2)))


@pytest.mark.parametrize("kind", HEAD_DIMS)
def test_strided_out(kind):
    q, k, v = _batch(kind, 2, 500, 700, 8, 2)
    want = _fwd(q, k, v, causal=True)
    buf = torch.full((2, 8, 500 + 3, 128 + 32), float("nan"), device="cuda").bfloat16()
    out = buf[:, :, 3:, 32:].transpose(1, 2)  # BSHD view into a padded BHSD buffer
    got = _fwd(q, k, v, causal=True, out=out)
    assert got.data_ptr() == out.data_ptr() and torch.equal(_bits(out), _bits(want))
    assert buf[:, :, :3].isnan().all() and buf[..., :32].isnan().all(), (
        "wrote outside out"
    )


def test_expanded_kv_heads():
    """k/v heads broadcast with expand (head stride 0) count as MQA."""
    q, k, v = _batch("d128", 2, 500, 700, 8, 1)
    got = _fwd(q, k.expand(-1, -1, 8, -1), v.expand(-1, -1, 8, -1), causal=True)
    for b in range(2):
        _assert_close(got[b], _reference(q[b], k[b], v[b], True, 128**-0.5))


@pytest.mark.parametrize("kind", HEAD_DIMS)
@pytest.mark.parametrize("how", ["head-major", "padded", "packed qkv"])
def test_varlen_strided_equals_contiguous_bitwise(kind, how):
    sq, skv = [385, 0, 47, 1000], [1000, 64, 47, 1000]
    H, HKv = 8, 2
    dqk, dv = HEAD_DIMS[kind]
    g = torch.Generator(device="cuda").manual_seed(9)
    Tq, Tk = sum(sq), sum(skv)
    if how == "head-major":  # [heads, tokens, d] buffer layouts
        q = _randn(g, H, Tq, dqk).transpose(0, 1)
        k = _randn(g, HKv, Tk, dqk).transpose(0, 1)
        v = _randn(g, HKv, Tk, dv).transpose(0, 1)
    elif how == "padded":
        q = _randn(g, Tq, H + 1, dqk + 64)[:, 1:, :dqk]
        k = _randn(g, Tk, HKv, dqk + 8)[..., 8:]
        v = _randn(g, Tk, HKv + 2, dv)[:, :HKv]
    else:  # one fused projection: [tokens, q + k + v heads, d] (self-attention lengths)
        sq = skv = [385, 0, 47, 1000]
        Tq = Tk = sum(sq)
        d = max(dqk, dv)
        qkv = _randn(g, Tq, H + 2 * HKv, d)
        q, k, v = qkv[:, :H, :dqk], qkv[:, H : H + HKv, :dqk], qkv[:, H + HKv :, :dv]
    args = (_cu(sq), _cu(skv), max(sq), max(skv))
    want = ma.forward_varlen(
        q.contiguous(), k.contiguous(), v.contiguous(), *args, causal=True
    )
    got = ma.forward_varlen(q, k, v, *args, causal=True)
    assert torch.equal(_bits(got), _bits(want))
    _check_varlen(q, k, v, sq, skv, True, got)
    # an output view in head-major order
    buf = torch.empty(H, Tq, dv, device="cuda").bfloat16()
    got = ma.forward_varlen(q, k, v, *args, causal=True, out=buf.transpose(0, 1))
    assert torch.equal(_bits(got), _bits(want))


# ---- call conventions -----------------------------------------------------------------------------


@pytest.mark.parametrize("kind", HEAD_DIMS)
def test_out_argument(kind):
    q, k, v = _batch(kind, 2, 500, 700, 8, 2)
    want = _fwd(q, k, v, causal=True)
    out = torch.full_like(want, float("nan"))
    got = _fwd(q, k, v, causal=True, out=out)
    assert got.data_ptr() == out.data_ptr() and torch.equal(_bits(out), _bits(want))

    qp, kp, vp = q.flatten(0, 1), k.flatten(0, 1), v.flatten(0, 1)
    out = torch.full_like(want.flatten(0, 1), float("nan"))
    got = ma.forward_varlen(
        qp, kp, vp, _cu([500, 500]), _cu([700, 700]), 500, 700, causal=True, out=out
    )
    assert got.data_ptr() == out.data_ptr() and torch.equal(
        _bits(out), _bits(want.flatten(0, 1))
    )


def test_deterministic_and_rtne_is_the_default():
    q, k, v = _batch("d128", 2, 1000, 3000, 8, 2)
    a = _fwd(q, k, v, causal=True)
    b = ma.forward(q, k, v, causal=True, round_mode="rtne", layout="bshd")
    assert torch.equal(_bits(a), _bits(b))


def test_runs_on_the_current_stream_and_in_graphs():
    q, k, v = _batch("d128", 2, 1000, 3000, 8, 2)
    want = _fwd(q, k, v, causal=True)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        got = _fwd(q, k, v, causal=True)
    side.synchronize()
    assert torch.equal(_bits(got), _bits(want))

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = _fwd(q, k, v, causal=True)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(_bits(out), _bits(want))


def test_cpu_tensors_round_trip():
    q, k, v = _batch("d128", 1, 300, 400, 4, 2)
    got = _fwd(q.cpu(), k.cpu(), v.cpu(), causal=True)
    assert got.device.type == "cpu"
    assert torch.equal(_bits(got), _bits(_fwd(q, k, v, causal=True).cpu()))


# ---- invalid input handling -----------------------------------------------------------------------


def _t(*shape, dtype=torch.bfloat16):
    return torch.zeros(*shape, dtype=dtype, device="cuda")


def _i32(*values):
    return torch.tensor(values, dtype=torch.int32)


@pytest.fixture
def t():
    """Valid inputs to break one at a time: a batch (q, kv) and packed (pq, pkv), each with cu = [0, 64]."""
    return types.SimpleNamespace(
        q=_t(1, 64, 4, 128),
        kv=_t(1, 64, 2, 128),
        pq=_t(64, 4, 128),
        pkv=_t(64, 2, 128),
        cu=_i32(0, 64),
    )


INVALID = {
    "fp16": (TypeError, lambda t: _fwd(t.q.half(), t.kv, t.kv)),
    "3-D batch": (ValueError, lambda t: _fwd(t.pq, t.pkv, t.pkv)),
    "head dim 64": (
        ValueError,
        lambda t: _fwd(_t(1, 64, 4, 64), _t(1, 64, 2, 64), _t(1, 64, 2, 64)),
    ),
    "head dims 192/192": (
        ValueError,
        lambda t: _fwd(_t(1, 64, 4, 192), _t(1, 64, 2, 192), _t(1, 64, 2, 192)),
    ),
    "k head dim": (ValueError, lambda t: _fwd(_t(1, 64, 4, 192), t.kv, t.kv)),
    "heads not a multiple": (
        ValueError,
        lambda t: _fwd(_t(1, 64, 3, 128), t.kv, t.kv),
    ),
    "k/v heads differ": (
        ValueError,
        lambda t: _fwd(t.q, t.kv, _t(1, 64, 4, 128)),
    ),
    "batch differs": (ValueError, lambda t: _fwd(_t(2, 64, 4, 128), t.kv, t.kv)),
    "k/v lengths differ": (
        ValueError,
        lambda t: _fwd(t.q, t.kv, _t(1, 65, 2, 128)),
    ),
    "head dim strided": (
        ValueError,
        lambda t: _fwd(_t(1, 64, 4, 256)[..., ::2], t.kv, t.kv),
    ),
    "out head dim strided": (
        ValueError,
        lambda t: _fwd(t.q, t.kv, t.kv, out=_t(1, 64, 4, 256)[..., ::2]),
    ),
    "varlen head dim strided": (
        ValueError,
        lambda t: ma.forward_varlen(
            t.pq, _t(64, 2, 128, 2)[..., 0], t.pkv, t.cu, t.cu, 64, 64
        ),
    ),
    "token stride > 2^22": (
        ValueError,
        lambda t: _fwd(
            _t(2**22 + 1 + 4 * 128).as_strided((1, 2, 4, 128), (0, 2**22 + 1, 128, 1)),
            t.kv[:, :2],
            t.kv[:, :2],
        ),
    ),
    "out shape": (
        ValueError,
        lambda t: _fwd(t.q, t.kv, t.kv, out=_t(1, 64, 4, 64)),
    ),
    "out dtype": (
        ValueError,
        lambda t: _fwd(t.q, t.kv, t.kv, out=_t(1, 64, 4, 128, dtype=torch.float32)),
    ),
    "scale 0": (ValueError, lambda t: _fwd(t.q, t.kv, t.kv, softmax_scale=0.0)),
    "scale nan": (
        ValueError,
        lambda t: _fwd(t.q, t.kv, t.kv, softmax_scale=float("nan")),
    ),
    "scale inf": (
        ValueError,
        lambda t: _fwd(t.q, t.kv, t.kv, softmax_scale=float("inf")),
    ),
    "bhsd k/v lengths differ": (
        ValueError,
        lambda t: ma.forward(t.q, _t(1, 64, 3, 128), t.kv),
    ),
    "layout": (ValueError, lambda t: ma.forward(t.q, t.kv, t.kv, layout="sbhd")),
    "rtna": (ValueError, lambda t: _fwd(t.q, t.kv, t.kv, round_mode="rtna")),
    "rtz": (ValueError, lambda t: _fwd(t.q, t.kv, t.kv, round_mode="rtz")),
    "varlen 4-D": (
        ValueError,
        lambda t: ma.forward_varlen(t.q, t.kv, t.kv, t.cu, t.cu, 64, 64),
    ),
    "varlen cpu q": (
        ValueError,
        lambda t: ma.forward_varlen(t.pq.cpu(), t.pkv, t.pkv, t.cu, t.cu, 64, 64),
    ),
    "cu int64": (
        ValueError,
        lambda t: ma.forward_varlen(t.pq, t.pkv, t.pkv, t.cu.long(), t.cu, 64, 64),
    ),
    "cu 2-D": (
        ValueError,
        lambda t: ma.forward_varlen(t.pq, t.pkv, t.pkv, t.cu[None], t.cu, 64, 64),
    ),
    "cu empty": (
        ValueError,
        lambda t: ma.forward_varlen(t.pq, t.pkv, t.pkv, t.cu[:0], t.cu[:0], 64, 64),
    ),
    "cu decreasing": (
        ValueError,
        lambda t: ma.forward_varlen(
            t.pq, t.pkv, t.pkv, _i32(0, 40, 30), _i32(0, 30, 40), 64, 64
        ),
    ),
    "cu negative": (
        ValueError,
        lambda t: ma.forward_varlen(t.pq, t.pkv, t.pkv, _i32(-1, 64), t.cu, 64, 64),
    ),
    "cu past tokens": (
        ValueError,
        lambda t: ma.forward_varlen(t.pq, t.pkv, t.pkv, _i32(0, 65), t.cu, 64, 64),
    ),
    "cu lengths differ": (
        ValueError,
        lambda t: ma.forward_varlen(t.pq, t.pkv, t.pkv, t.cu, _i32(0, 32, 64), 64, 64),
    ),
    "varlen k/v tokens differ": (
        ValueError,
        lambda t: ma.forward_varlen(t.pq, t.pkv, _t(65, 2, 128), t.cu, t.cu, 64, 64),
    ),
    "cu past max_seqlen": (
        ValueError,
        lambda t: ma.forward_varlen(t.pq, t.pkv, t.pkv, t.cu, t.cu, 32, 64),
    ),
    "gpu cu int64": (
        ValueError,
        lambda t: ma.forward_varlen(
            t.pq, t.pkv, t.pkv, t.cu.cuda().long(), t.cu.cuda(), 64, 64
        ),
    ),
    "cu on two devices": (
        ValueError,
        lambda t: ma.forward_varlen(t.pq, t.pkv, t.pkv, t.cu, t.cu.cuda(), 64, 64),
    ),
    "host copies of host cu": (
        ValueError,
        lambda t: ma.forward_varlen(
            t.pq,
            t.pkv,
            t.pkv,
            t.cu,
            t.cu,
            64,
            64,
            host_cu_seqlens_q=t.cu,
            host_cu_seqlens_k=t.cu,
        ),
    ),
    "host copy of q only": (
        ValueError,
        lambda t: ma.forward_varlen(
            t.pq, t.pkv, t.pkv, t.cu.cuda(), t.cu.cuda(), 64, 64, host_cu_seqlens_q=t.cu
        ),
    ),
    "host copies on the gpu": (
        ValueError,
        lambda t: ma.forward_varlen(
            t.pq,
            t.pkv,
            t.pkv,
            t.cu.cuda(),
            t.cu.cuda(),
            64,
            64,
            host_cu_seqlens_q=t.cu.cuda(),
            host_cu_seqlens_k=t.cu.cuda(),
        ),
    ),
    "host copies past max_seqlen": (
        ValueError,
        lambda t: ma.forward_varlen(
            t.pq,
            t.pkv,
            t.pkv,
            t.cu.cuda(),
            t.cu.cuda(),
            64,
            32,
            host_cu_seqlens_q=t.cu,
            host_cu_seqlens_k=t.cu,
        ),
    ),
    "varlen out shape": (
        ValueError,
        lambda t: ma.forward_varlen(
            t.pq, t.pkv, t.pkv, t.cu, t.cu, 64, 64, out=_t(64, 4, 64)
        ),
    ),
}


@pytest.mark.parametrize("case", INVALID)
def test_rejects_invalid_inputs(t, case):
    error, call = INVALID[case]
    with pytest.raises(error):
        call(t)


def _plan_t(t, **kw):
    """t's packed call (cu [0, 64], 4 / 2 heads, D128) planned, with kw overriding arguments."""
    a = dict(
        cu_seqlens_q=t.cu.cuda(),
        cu_seqlens_k=t.cu.cuda(),
        max_seqlen_q=64,
        max_seqlen_k=64,
        total_q=64,
        total_k=64,
        heads=4,
        kv_heads=2,
        head_dims=(128, 128),
        causal=True,
    )
    return ma.VarlenPlan().plan(**{**a, **kw})


INVALID_PLAN = {
    "host cu": (ValueError, lambda t: _plan_t(t, cu_seqlens_q=t.cu, cu_seqlens_k=t.cu)),
    "head dims 64": (ValueError, lambda t: _plan_t(t, head_dims=(64, 64))),
    "heads not a multiple": (ValueError, lambda t: _plan_t(t, heads=3)),
    "host copy of k only": (ValueError, lambda t: _plan_t(t, host_cu_seqlens_k=t.cu)),
    "host copies past total": (
        ValueError,
        lambda t: _plan_t(
            t, total_q=32, host_cu_seqlens_q=t.cu, host_cu_seqlens_k=t.cu
        ),
    ),
    "run before plan": (
        RuntimeError,
        lambda t: ma.VarlenPlan().run(t.pq, t.pkv, t.pkv),
    ),
    "run other tokens": (ValueError, lambda t: _plan_t(t).run(t.pq[:63], t.pkv, t.pkv)),
    "run other heads": (
        ValueError,
        lambda t: _plan_t(t).run(_t(64, 8, 128), t.pkv, t.pkv),
    ),
    "run other head dims": (
        ValueError,
        lambda t: _plan_t(t).run(_t(64, 4, 192), _t(64, 2, 192), t.pkv),
    ),
    "run after close": (
        RuntimeError,
        lambda t: (lambda p: (p.close(), p.run(t.pq, t.pkv, t.pkv)))(_plan_t(t)),
    ),
}


@pytest.mark.parametrize("case", INVALID_PLAN)
def test_plan_rejects_invalid_inputs(t, case):
    error, call = INVALID_PLAN[case]
    with pytest.raises(error):
        call(t)


def _paged_t(t, page=16, **kw):
    """t's packed call (cu [0, 64], 4 / 2 heads) on `page`-token page caches, with kw overriding fields."""
    pages = -(-64 // page)
    a = dict(
        q=t.pq,
        k_cache=_t(pages, page, 2, 128),
        v_cache=_t(pages, page, 2, 128),
        cu_seqlens_q=t.cu,
        seqlens_k=_i32(64),
        kv_indptr=_i32(0, pages),
        kv_page_indices=torch.arange(pages, dtype=torch.int32, device="cuda"),
    )
    a.update(kw)
    return ma.forward_paged(**a)


INVALID_PAGED = {
    "page 4": (ValueError, lambda t: _paged_t(t, page=4)),
    "page 24": (ValueError, lambda t: _paged_t(t, page=24)),
    "mla page 8": (
        ValueError,
        lambda t: _paged_t(
            t,
            page=8,
            q=_t(64, 4, 192),
            k_cache=_t(8, 8, 2, 192),
            v_cache=_t(8, 8, 2, 128),
        ),
    ),
    "table too short": (ValueError, lambda t: _paged_t(t, seqlens_k=_i32(65))),
    "k and v strides differ": (
        ValueError,
        lambda t: _paged_t(t, v_cache=_t(4, 2, 16, 128).transpose(1, 2)),
    ),
    "both page lists": (
        ValueError,
        lambda t: _paged_t(
            t, block_table=torch.zeros(1, 4, dtype=torch.int32, device="cuda")
        ),
    ),
    "no page list": (
        ValueError,
        lambda t: _paged_t(t, kv_indptr=None, kv_page_indices=None),
    ),
    "int64 page table": (
        ValueError,
        lambda t: _paged_t(t, kv_page_indices=torch.arange(4, device="cuda")),
    ),
}


@pytest.mark.parametrize("case", INVALID_PAGED)
def test_paged_rejects_invalid_inputs(t, case):
    error, call = INVALID_PAGED[case]
    with pytest.raises(error):
        call(t)


@pytest.mark.parametrize(
    "op",
    [
        "forward_lite",
        "mla_decode_a16w8",
        "mla_dcp_lse_merge_ranks",
        "mxfp4_moe_gateup",
        "mxfp4_moe_down",
        "mxfp4_moe_gateup_block_m",
    ],
)
def test_cdna3_ops_say_gfx942_only(op):
    fn = getattr(ma, op)
    params = inspect.signature(fn).parameters.values()
    args = [
        None
        for p in params
        if p.default is p.empty and p.kind is p.POSITIONAL_OR_KEYWORD
    ]
    kwargs = {
        p.name: None
        for p in params
        if p.default is p.empty and p.kind is p.KEYWORD_ONLY
    }
    with pytest.raises(RuntimeError, match="gfx942"):
        fn(*args, **kwargs)
