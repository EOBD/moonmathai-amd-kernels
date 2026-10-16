"""Sanity-check moonmath_amd against SDPA.

Shapes mirror runner.py (B=2, H=24, S=16384, D=128) and the README's
cross-attention example (KV=512).
"""

from collections import namedtuple

import pytest
import torch
import torch.nn.functional as F

import moonmath_amd as ma
from moonmath_amd import _arch

# gfx942-only kernels: skip when this build lacks the module or the GPU is not CDNA3.
pytestmark = pytest.mark.skipif(
    not _arch.is_built("gfx942") or (torch.cuda.is_available() and _arch.get_gfx_runtime() != "gfx942"),
    reason="needs moonmath_amd._C_gfx942 and a gfx942 GPU",
)

LAYOUTS = ("bshd", "bhsd")

# Test shapes as (B, H, S, D); S_kv is the cross-attention KV length.
# Kernel constraints: D == 128, S % 256 == 0, and S_kv % 64 == 0.
Shape = namedtuple("Shape", "B H S D S_kv")
SHAPES = {
    "small": Shape(1, 4, 1024, 128, 256),
    "medium": Shape(2, 8, 4096, 128, 512),
    "runner": Shape(2, 24, 16384, 128, 512),
}


@pytest.fixture(scope="session")
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA/HIP device not available")
    return torch.device("cuda")


def _randn(B, S, H, D, device):
    return torch.randn(B, S, H, D, dtype=torch.bfloat16, device=device)


def _rand_self_bshd(shape, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(42)
    q = _randn(shape.B, shape.S, shape.H, shape.D, device)
    k = _randn(shape.B, shape.S, shape.H, shape.D, device)
    v = _randn(shape.B, shape.S, shape.H, shape.D, device)
    return q, k, v


def _rand_cross_bshd(shape, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(42)
    q = _randn(shape.B, shape.S, shape.H, shape.D, device)
    k = _randn(shape.B, shape.S_kv, shape.H, shape.D, device)
    v = _randn(shape.B, shape.S_kv, shape.H, shape.D, device)
    return q, k, v


def _as_layout(q, k, v, layout):
    """Convert the BSHD base tensors to the requested `layout`."""
    if layout == "bshd":
        return q, k, v
    return tuple(t.transpose(1, 2).contiguous() for t in (q, k, v))


def _sdpa(q, k, v, layout):
    """Run SDPA (which wants BHSD), converting from `layout` and back."""
    if layout == "bshd":
        q, k, v = (t.transpose(1, 2).contiguous() for t in (q, k, v))
    out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    return out.transpose(1, 2).contiguous() if layout == "bshd" else out


@pytest.mark.gpu
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("shape", SHAPES.values(), ids=SHAPES.keys())
def test_self_attention_sdpa_sanity(device, shape, layout):
    q, k, v = _as_layout(*_rand_self_bshd(shape, device), layout)
    out = ma.forward(q, k, v, round_mode="rtna", layout=layout)
    ref = _sdpa(q, k, v, layout)
    torch.testing.assert_close(out.float(), ref.float(), rtol=0.05, atol=0.05)


@pytest.mark.gpu
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("shape", SHAPES.values(), ids=SHAPES.keys())
def test_cross_attention_sdpa_sanity(device, shape, layout):
    q, k, v = _as_layout(*_rand_cross_bshd(shape, device), layout)
    out = ma.forward(q, k, v, round_mode="rtna", layout=layout)
    ref = _sdpa(q, k, v, layout)
    torch.testing.assert_close(out.float(), ref.float(), rtol=0.05, atol=0.05)


# ---- LiteAttention (cross-timestep skip + LPT scheduling) -------------------
#
# The dense suite above never exercises the lite kernel, the LPT work-balance
# planner, or the must_do_list force-compute path, and each can regress on its
# own, so a dedicated test pins each:
#   - kernel math:   the compute-all seed step must match full attention (SDPA);
#   - LPT scheduler: dispatching q-blocks heaviest-first must be BIT-identical to
#                    plain identity order (only the dispatch order changes);
#   - must_do_list:  forcing every K-block to "compute" must reproduce the
#                    compute-all result at each step (nothing is ever skipped).

LITE_SHAPE = Shape(1, 8, 4096, 128, 4096)  # satisfies S_kv % 64 == 0 and D == 128
LITE_STEPS = 5
LITE_THRESHOLD = -3.0


def _lite_qkv_uniform(device, layout):
    """Well-conditioned random self-attn inputs, for the compute-all-versus-SDPA check."""
    torch.manual_seed(7)
    q = _randn(LITE_SHAPE.B, LITE_SHAPE.S, LITE_SHAPE.H, LITE_SHAPE.D, device)
    k = _randn(LITE_SHAPE.B, LITE_SHAPE.S, LITE_SHAPE.H, LITE_SHAPE.D, device)
    v = _randn(LITE_SHAPE.B, LITE_SHAPE.S, LITE_SHAPE.H, LITE_SHAPE.D, device)
    return _as_layout(q, k, v, layout)


def _lite_qkv_peaked(device, layout):
    """Peaked inputs that force genuine, head-varying skips (for the bit-identity tests).

    At this threshold uniform-random K never skips: every block scores within thr
    of the max. Head h therefore keeps K-blocks 0..h at full scale and damps the
    rest to ~0, and the vote drops the cold blocks. Per-CTA work then spans 1..H
    kept blocks across the grid, so the LPT planner actually reorders instead of
    no-op'ing. These tests compare lite against lite, so the bf16 conditioning of
    the damped values is irrelevant.
    """
    torch.manual_seed(7)
    B, S, H, D = LITE_SHAPE.B, LITE_SHAPE.S, LITE_SHAPE.H, LITE_SHAPE.D
    q = torch.randn(B, S, H, D, device=device)
    k = torch.randn(B, S, H, D, device=device)
    v = torch.randn(B, S, H, D, device=device)
    scale = torch.full((H, S), 0.05, device=device)
    for h in range(H):
        scale[h, : (h + 1) * 64] = 2.0  # head h keeps blocks 0..h at full scale
    k = k * scale.t().view(1, S, H, 1)
    q, k, v = (t.to(torch.bfloat16) for t in (q, k, v))
    return _as_layout(q, k, v, layout)


def _run_lite(q, k, v, layout, steps, *, must_do_list=None):
    """Drive one LiteAttention through `steps` denoising steps; return the final output."""
    attn = ma.LiteAttention(threshold=LITE_THRESHOLD, round_mode="rtna", layout=layout)
    out = None
    for _ in range(steps):
        out = attn(q, k, v, must_do_list=must_do_list)
    return out


def _kept_blocks(skip_row):
    """Count K-blocks kept by one CTA, decoded from its int16 RLE row [len, hi0, lo0, ...].
    Order-independent: each [hi, lo] pair contributes |hi - lo| blocks."""
    r = skip_row.tolist()
    n = int(r[0])
    return sum(abs(int(r[i]) - int(r[i - 1])) for i in range(n, 1, -2))


@pytest.mark.gpu
@pytest.mark.parametrize("layout", LAYOUTS)
def test_lite_compute_all_matches_sdpa(device, layout):
    """The first lite step (the compute-all seed) is exactly full attention."""
    q, k, v = _lite_qkv_uniform(device, layout)
    attn = ma.LiteAttention(threshold=LITE_THRESHOLD, round_mode="rtna", layout=layout)
    out = attn(q, k, v)  # seed step: nothing skipped
    ref = _sdpa(q, k, v, layout)
    torch.testing.assert_close(out.float(), ref.float(), rtol=0.05, atol=0.05)


@pytest.mark.gpu
def test_lite_lpt_matches_plain_bitexact(device, monkeypatch):
    """Reordering q-blocks for LPT only changes dispatch order, so output is bit-identical."""
    q, k, v = _lite_qkv_peaked(device, "bshd")

    monkeypatch.delenv("MOONMATH_LITE_PLAIN", raising=False)  # default: LPT
    lpt = _run_lite(q, k, v, "bshd", LITE_STEPS)

    monkeypatch.setenv("MOONMATH_LITE_PLAIN", "1")  # force plain identity order
    plain = _run_lite(q, k, v, "bshd", LITE_STEPS)

    assert torch.equal(lpt, plain), "LPT reorder changed results (must be bit-identical)"


@pytest.mark.gpu
def test_lite_must_do_all_equals_compute_all(device):
    """Pinning every K-block with must_do_list forces compute-all at every step."""
    q, k, v = _lite_qkv_peaked(device, "bshd")
    ktiles = (LITE_SHAPE.S_kv + 63) // 64
    # RLE [len, start0, end0] = one range [0, ktiles), i.e. every K-block.
    must_do_all = torch.tensor([2, 0, ktiles], dtype=torch.int16, device=device)

    ref = ma.LiteAttention(threshold=LITE_THRESHOLD, round_mode="rtna", layout="bshd")(
        q, k, v
    )  # the compute-all seed
    forced = _run_lite(q, k, v, "bshd", LITE_STEPS, must_do_list=must_do_all)
    assert torch.equal(forced, ref), "forcing all K-blocks did not reproduce compute-all"


@pytest.mark.gpu
def test_lite_skip_list_develops(device):
    """The vote learns a skip list that genuinely drops K-blocks as steps advance.

    In bf16 skipping is numerically lossless -- a block is voted out only once its
    softmax weight underflows to ~0 -- so the output cannot reveal it, and we read
    the learned skip list directly. The data makes head h attend K-blocks 0..h, so
    a correct data-driven vote keeps exactly h+1 blocks; this proves skips occur,
    the list evolves from the compute-all seed, and the *right* blocks survive
    (per head, not a blanket cut)."""
    q, k, v = _lite_qkv_peaked(device, "bshd")
    ktiles = LITE_SHAPE.S_kv // 64
    attn = ma.LiteAttention(threshold=LITE_THRESHOLD, round_mode="rtna", layout="bshd")

    attn(q, k, v)  # seed step: reads compute-all and writes the first learned list
    assert _kept_blocks(attn.write_list[0, 0, 0]) == ktiles  # seed consumed every block
    assert _kept_blocks(attn.read_list[0, 0, 0]) < ktiles  # the learned list dropped some blocks

    for _ in range(LITE_STEPS - 1):
        attn(q, k, v)
    kept_by_head = [_kept_blocks(attn.read_list[0, h, 0]) for h in range(LITE_SHAPE.H)]
    assert kept_by_head == list(range(1, LITE_SHAPE.H + 1)), kept_by_head
