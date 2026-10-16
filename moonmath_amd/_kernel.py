"""ma.forward, forward_varlen, forward_paged and forward_lite, dispatched to the live GPU's module.

gfx942 (CDNA3): moonmath_amd._C_gfx942. gfx950 (CDNA4): moonmath_amd._C_gfx950.
"""

import math

import torch

from . import _arch


def forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    round_mode: str | None = None,
    layout: str = "bhsd",
    causal: bool = False,
    softmax_scale: float | None = None,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Fused forward attention: O = softmax(QKᵀ · softmax_scale) V, using the live GPU's kernels.

    Args:
        q, k, v: torch.bfloat16 tensors shaped (batch, heads, seq_len, head_dim)
                 or (batch, seq_len, heads, head_dim), depending on layout.
                 gfx942: contiguous. gfx950: any strides provided head_dim is contiguous (stride(-1) == 1):
                 views, slices, padded buffers, each tensor in its own memory order.
                 CPU tensors are moved to the GPU under the hood; GPU tensors are used in place.
        out: optional preallocated output, q's shape with v's head_dim, same dtype and
             device; contiguous on gfx942, any strides as above on gfx950. When None, a new
             contiguous tensor is allocated.
        round_mode: fp32 -> bf16 output rounding.
            gfx942: "rtna" (default; ties-away-from-zero), "rtne" (~0.5 ULP error), or "rtz"
            (cheapest; ~1 ULP, truncation).
            gfx950: "rtne" only (the hardware conversion), and the default.
        layout: "bhsd" (default; batch, heads, seq, head_dim) or "bshd" (batch, seq, heads,
            head_dim): how dims are read, on every arch. gfx950 reads either through strides,
            with no transposes.
        causal: bottom-right aligned causal mask (key j visible to row i iff j <= i + Skv - Sq);
            rows that see no key are 0. gfx950 only.
        softmax_scale: defaults to 1/sqrt(head_dim) (the only value gfx942 accepts).
        return_lse: also return each q row's log-sum-exp (gfx950 only): fp32 [batch, heads, seq_len_q]
            (AITER's layout, either layout), the natural log of sum_j exp(q.k_j * softmax_scale);
            -inf for a row that sees no key.

    Head dims: gfx942 accepts 128 with k/v heads == q heads. gfx950 accepts (q/k, v) head dims
    (128, 128) or (192, 128) (MLA prefill), and GQA: q heads a multiple of k/v heads.

    Returns:
        torch.bfloat16 tensor of q's shape (last dim v's head_dim) on q's device; with return_lse,
        (out, lse).
    """
    # Type and basic validation, ahead of the device check
    for name, t in (("q", q), ("k", k), ("v", v)):
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"{name} must be torch.Tensor; got {type(t).__name__}")
        if t.dtype != torch.bfloat16:
            raise TypeError(f"{name} must be torch.bfloat16; got dtype={t.dtype}")
        if t.dim() != 4:
            raise ValueError(
                f"{name} must be 4-D ({str(layout).upper()}); got shape={tuple(t.shape)}"
            )

    # CPU path: copy to the GPU, run, copy the result back
    if q.device.type == "cpu":
        # k and v must be on the CPU too
        if k.device.type != "cpu" or v.device.type != "cpu":
            raise ValueError(
                "q is on CPU but k or v is not; all must be on the same device"
            )

        # use the first available CUDA device
        device = torch.device("cuda:0")
        q_gpu = q.to(device)
        k_gpu = k.to(device)
        v_gpu = v.to(device)
        out_gpu = out.to(device) if out is not None else None

        result = _forward(
            q_gpu,
            k_gpu,
            v_gpu,
            out_gpu,
            round_mode,
            layout,
            causal,
            softmax_scale,
            return_lse,
        )
        if return_lse:
            return result[0].cpu(), result[1].cpu()
        return result.cpu()

    # GPU path: operate in place
    elif q.device.type in ("cuda", "hip"):
        # k and v must share q's device
        if k.device != q.device or v.device != q.device:
            raise ValueError(
                f"q/k/v must be on the same device; got q={q.device}, k={k.device}, v={v.device}"
            )

        return _forward(
            q, k, v, out, round_mode, layout, causal, softmax_scale, return_lse
        )

    else:
        raise NotImplementedError(
            f"Unsupported device {q.device!r}; expected 'cpu' or 'cuda'/'hip'"
        )


def _result(out, lse, return_lse):
    """Modules return (out, lse or None); ops return out, or (out, lse) when return_lse is set."""
    return (out, lse) if return_lse else out


def _forward(q, k, v, out, round_mode, layout, causal, softmax_scale, return_lse):
    arch = _arch.get_gfx_runtime(q.device)
    if layout not in ("bhsd", "bshd"):
        raise ValueError(f"layout must be 'bhsd' or 'bshd', got {layout!r}")
    if arch == "gfx950":
        if (round_mode or "rtne") != "rtne":
            raise ValueError(
                "gfx950 converts to bf16 in hardware with round-to-nearest-even: "
                f"round_mode must be 'rtne' (or unset), got {round_mode!r}"
            )
        o, lse = _arch.load_module("gfx950").forward(
            q, k, v, out, bool(causal), softmax_scale, layout, bool(return_lse)
        )
        return _result(o, lse, return_lse)

    if arch != "gfx942":
        raise RuntimeError(
            f"moonmath_amd.forward: no kernels for {arch}; supported: gfx942, gfx950"
        )
    if causal:
        raise NotImplementedError(
            "gfx942: the forward kernel is dense only; causal needs gfx950"
        )
    if return_lse:
        raise NotImplementedError("gfx942: return_lse needs gfx950")
    if softmax_scale is not None and not math.isclose(
        softmax_scale, q.shape[-1] ** -0.5, rel_tol=1e-6
    ):
        raise NotImplementedError("gfx942: softmax_scale is fixed at 1/sqrt(head_dim)")
    return _arch.load_module("gfx942").forward(
        q, k, v, out, round_mode or "rtna", layout
    )


def forward_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    *,
    causal: bool = False,
    softmax_scale: float | None = None,
    out: torch.Tensor | None = None,
    return_lse: bool = False,
    host_cu_seqlens_q: torch.Tensor | None = None,
    host_cu_seqlens_k: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Variable-length attention over packed sequences, matching flash_attn_varlen_func's argument order. gfx950 only.

    Args:
        q: [total_q, heads, head_dim_qk], k: [total_k, kv_heads, head_dim_qk], v: [total_k, kv_heads,
            head_dim_v]; bf16, on the GPU, any token and head strides with a contiguous head_dim
            (e.g. a head-major [heads, total, head_dim] buffer's .transpose(0, 1)). Head dims (128, 128)
            or (192, 128); GQA when heads % kv_heads == 0.
        cu_seqlens_q, cu_seqlens_k: int32 [num_seqs + 1]: sequence i is q tokens
            cu_seqlens_q[i] .. cu_seqlens_q[i + 1] - 1, and k/v tokens likewise; zero lengths allowed.
            On the GPU: planned on the GPU, so no host sync and CUDA-graph capturable (a replay reads
            the arrays' current contents). On the host (CPU tensors): planned exactly on the host.
        max_seqlen_q, max_seqlen_k: at least the longest q and k/v sequence; they size a GPU plan.
        causal: bottom-right aligned within each sequence; rows that see no key are 0.
        softmax_scale: defaults to 1/sqrt(head_dim_qk).
        out: optional preallocated [total_q, heads, head_dim_v] bf16 output (strides as q).
        return_lse: also return lse, fp32 [heads, total_q] (AITER's varlen layout): each row's natural
            log-sum-exp of its scaled scores, -inf for a row that sees no key.
        host_cu_seqlens_q, host_cu_seqlens_k: optional CPU copies of GPU cu_seqlens: an eager call then
            plans exactly on the host (a graph capture ignores them: a replay may change the lengths).

    Rows outside every sequence are not written (out and lse). To plan once for many layers: VarlenPlan.

    Returns:
        torch.bfloat16 tensor [total_q, heads, head_dim_v]; with return_lse, (out, lse).
    """
    _require_gfx950("forward_varlen", q)
    o, lse = _arch.load_module("gfx950").forward_varlen(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        int(max_seqlen_q),
        int(max_seqlen_k),
        out,
        bool(causal),
        softmax_scale,
        bool(return_lse),
        host_cu_seqlens_q,
        host_cu_seqlens_k,
    )
    return _result(o, lse, return_lse)


def forward_paged(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqlens_k: torch.Tensor,
    *,
    kv_indptr: torch.Tensor | None = None,
    kv_page_indices: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
    causal: bool = False,
    softmax_scale: float | None = None,
    out: torch.Tensor | None = None,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Variable-length attention with K/V held in pages, as AITER's mha_batch_prefill_func. gfx950 only.

    Args:
        q: [total_q, heads, head_dim_qk], as forward_varlen's.
        k_cache, v_cache: [num_pages, page_size, kv_heads, head_dim], bf16, any strides with a contiguous
            head_dim (e.g. views of one [num_pages, 2, page_size, kv_heads, head_dim] cache). page_size a
            power of two: at least 8 for head dims (128, 128), 16 for (192, 128). Head dim 128: k_cache
            and v_cache must have the same strides.
        cu_seqlens_q: int32 [num_seqs + 1]: sequence i is q tokens cu_seqlens_q[i] .. cu_seqlens_q[i + 1] - 1.
        seqlens_k: int32 [num_seqs]: each sequence's key count (prefix and new tokens).
        kv_indptr, kv_page_indices: SGLang / AITER page lists: sequence i's pages are
            kv_page_indices[kv_indptr[i] .. kv_indptr[i + 1] - 1].
        block_table: or vLLM's int32 [num_seqs, max_pages]: sequence i's pages are block_table[i].
        causal, softmax_scale, out, return_lse: as forward_varlen's.

    cu_seqlens_q, seqlens_k and kv_indptr are read on the host: CPU tensors avoid a sync (GPU ones are
    copied). The page lists stay on the GPU; their entries must reference pages of the caches (not checked).

    Returns:
        torch.bfloat16 tensor [total_q, heads, head_dim_v]; with return_lse, (out, lse [heads, total_q]).
    """
    _require_gfx950("forward_paged", q)
    given = (
        kv_indptr is not None,
        kv_page_indices is not None,
        block_table is not None,
    )
    if given not in ((True, True, False), (False, False, True)):
        raise ValueError(
            "forward_paged: give kv_indptr and kv_page_indices, or block_table"
        )
    num_seqs = cu_seqlens_q.numel() - 1
    if block_table is not None:
        page_table = block_table.contiguous().view(-1)
        page_starts = torch.arange(num_seqs, dtype=torch.int32) * block_table.size(-1)
    else:
        page_table = kv_page_indices
        page_starts = _host_ints(kv_indptr)[:-1].contiguous()
    o, lse = _arch.load_module("gfx950").forward_paged(
        q,
        k_cache,
        v_cache,
        _host_ints(cu_seqlens_q),
        page_table,
        page_starts,
        _host_ints(seqlens_k),
        out,
        bool(causal),
        softmax_scale,
        bool(return_lse),
    )
    return _result(o, lse, return_lse)


def _host_ints(t):
    """Return t as a contiguous int32 CPU tensor (a GPU tensor is copied, inducing a sync)."""
    return t.to("cpu", torch.int32).contiguous()


def _require_gfx950(name, t):
    """Raise a clear error unless tensor t lives on a gfx950 GPU (checked before loading that module)."""
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"{name}: expected a torch.Tensor; got {type(t).__name__}")
    arch = _arch.get_gfx_runtime(t.device)
    if arch != "gfx950":
        raise RuntimeError(
            f"moonmath_amd.{name} runs on gfx950 (CDNA4) only; {t.device} is {arch}"
        )


class VarlenPlan:
    """forward_varlen planned once per forward pass, then run for every layer. gfx950 only.

    plan() takes forward_varlen's sequence arguments (GPU cu_seqlens, optional host copies) and the
    runs' shapes; run(q, k, v) then launches just the attention kernels. Both are CUDA-graph
    capturable; a run follows its plan within the same capture, or both outside a capture.
    """

    def __init__(self):
        self._plan = None  # the gfx950 module's VarlenPlan, created on the first plan()

    def plan(
        self,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        total_q: int,
        total_k: int,
        heads: int,
        kv_heads: int,
        head_dims: tuple[int, int] = (128, 128),
        causal: bool = False,
        *,
        host_cu_seqlens_q: torch.Tensor | None = None,
        host_cu_seqlens_k: torch.Tensor | None = None,
    ) -> "VarlenPlan":
        """Plan runs over q [total_q, heads, head_dims[0]], k/v [total_k, kv_heads, ...] (re-plan every pass)."""
        _require_gfx950("VarlenPlan", cu_seqlens_q)
        if self._plan is None:
            self._plan = _arch.load_module("gfx950").VarlenPlan()
        self._plan.plan(
            cu_seqlens_q,
            cu_seqlens_k,
            int(max_seqlen_q),
            int(max_seqlen_k),
            int(total_q),
            int(total_k),
            int(heads),
            int(kv_heads),
            tuple(head_dims),
            bool(causal),
            host_cu_seqlens_q,
            host_cu_seqlens_k,
        )
        return self

    def run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        softmax_scale: float | None = None,
        out: torch.Tensor | None = None,
        return_lse: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run forward_varlen(q, k, v, ...) on the planned sequences; causal comes from the plan."""
        if self._plan is None:
            raise RuntimeError("VarlenPlan.run: plan() it first")
        o, lse = self._plan.run(q, k, v, out, softmax_scale, bool(return_lse))
        return _result(o, lse, return_lse)

    def close(self):
        """Release the plan's GPU buffers after its queued kernels. A graph that captured it still needs
        them: keep the plan open while that graph may replay."""
        if self._plan is not None:
            self._plan.close()


def forward_lite(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    read_list: torch.Tensor,
    write_list: torch.Tensor,
    *,
    threshold: float,
    phase: int,
    must_do_list: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    round_mode: str = "rtna",
    layout: str = "bhsd",
) -> torch.Tensor:
    """LiteAttention forward: skip K-blocks per the cross-timestep skip list.

    `read_list` / `write_list` are int16 CUDA tensors shaped
    [B, H, qtiles, nblocks+2] (the per-phase slices of a [2, ...] double buffer):
    read_list = skip_list[phase], write_list = skip_list[1-phase]. The kernel
    processes only the K-blocks named in read_list and writes, for the next
    timestep, the blocks whose max score stayed within `threshold` (log2) of the
    running max for ALL 256 q-rows of the CTA. tile = (kBlockM=256, kBlockN=64).

    `must_do_list` (optional): a single GLOBAL int16 1-D tensor formatted as
    [len, tile_start0, tile_end0, tile_start1, tile_end1, ...] (end-exclusive
    K-block tile indices). Blocks in any [start, end) range are FORCED to be
    recorded as "compute" in write_list regardless of the skip vote — pinning
    them in every timestep's read list (e.g. attention sinks / always-attend
    blocks). None => no forcing.

    GPU tensors only (the diffusion use case). Output approximates full
    attention (skipped blocks contribute 0) — exact only with a compute-all list.

    Note: the lite kernel symbol is absent from the current attention_kernel.hip.
    This raises NotImplementedError until that kernel is added.
    """
    # currently raises from _C_gfx942.forward_lite
    return _arch.gfx942_op("forward_lite")(
        q,
        k,
        v,
        read_list,
        write_list,
        threshold,
        phase,
        must_do_list,
        out,
        round_mode,
        layout,
    )
