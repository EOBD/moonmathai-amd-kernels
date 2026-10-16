#!/usr/bin/env python3
"""moonmath_amd gfx950 attention vs every AITER bf16 forward kernel that runs the shape, on MI355X.

    python benchmark/bench_gfx950.py --markdown          # the README table
    python benchmark/bench_gfx950.py --only gqa --json out.json

AITER's gfx950 bf16 forward kernels, each timed wherever it accepts the shape:
  v4    aiter.ops.mha_v4.mha_v4: ASM, head dim 128, non-causal only
  v3    flash_attn_func's ASM v3 path (fwd_hd128_bf16*.co, fwd_hd192_hd128_bf16*.co); AITER picks it
        for head dim 128, and for 192/128 when AITER_DISABLE_FMHA_OPUS=1
  opus  flash_attn_func's OPUS path: AITER's default for 192/128, opt-in for 128 (AITER_ENABLE_FMHA_OPUS=1)
Each row is compared against the fastest of them.

Fairness rules:
  * identical bf16 inputs, BSHD contiguous (the only layout v4 accepts; moonmath reads it natively)
  * every kernel rounds the bf16 output RTNE (gfx950 AITER ships no other bf16_cvt variant)
  * neither side returns LSE; moonmath and v4 write a preallocated out, flash_attn_func allocates one per
    call (the caching allocator hands back the same block, eager and in a graph)
  * every kernel is checked against an fp32 reference before timing; a failing one is flagged in the table
  * timing = GPU events around a HIP graph of N back-to-back calls (no Python/launch overhead on either
    side); eager timing is kept as a cross-check
  * kernels interleaved within a pass, order rotated every pass, median over passes (clock/thermal drift)
  * everything a call launches (moonmath's KV-split merge, AITER's bookkeeping) is inside the timed region
"""

import argparse
import json
import math
import os
import statistics

import torch

import moonmath_amd as ma
from aiter import flash_attn_func
from aiter.ops.mha_v4 import AttentionFormat, mha_v4

BF16 = AttentionFormat.BF16
DEV = "cuda"
OPUS_ENV = ("AITER_ENABLE_FMHA_OPUS", "AITER_DISABLE_FMHA_OPUS")

# (label, B, Hq, Hkv, Sq, Skv, Dqk, Dv, causal)
SHAPES = [
    ("flux-1024px", 1, 24, 24, 4608, 4608, 128, 128, False),
    ("sd3-2x4096", 2, 24, 24, 4096, 4096, 128, 128, False),
    ("1x32x16k", 1, 32, 32, 16384, 16384, 128, 128, False),
    ("4x16x16k", 4, 16, 16, 16384, 16384, 128, 128, False),
    ("2x16x64k", 2, 16, 16, 65536, 65536, 128, 128, False),
    ("2x24x16k", 2, 24, 24, 16384, 16384, 128, 128, False),
    ("1x64x16k", 1, 64, 64, 16384, 16384, 128, 128, False),
    ("2x24x32k", 2, 24, 24, 32768, 32768, 128, 128, False),
    ("2x8x86k", 2, 8, 8, 86016, 86016, 128, 128, False),
    ("1x16x128k", 1, 16, 16, 131072, 131072, 128, 128, False),
    ("wan-480p", 1, 40, 40, 32760, 32760, 128, 128, False),
    ("wan-720p", 1, 40, 40, 75600, 75600, 128, 128, False),
    ("hunyuan-video", 1, 24, 24, 119056, 119056, 128, 128, False),
    ("odd-1x24x4097", 1, 24, 24, 4097, 4097, 128, 128, False),
    ("gqa8-1x64x8k", 1, 64, 8, 8192, 8192, 128, 128, False),
    ("gqa4-4x32x4k", 4, 32, 8, 4096, 4096, 128, 128, False),
    ("cross-2x24-8k/512", 2, 24, 24, 8192, 512, 128, 128, False),
    ("causal-1x32x8k", 1, 32, 32, 8192, 8192, 128, 128, True),
    ("causal-gqa8-1x64x16k", 1, 64, 8, 16384, 16384, 128, 128, True),
    ("causal-4x32x4k", 4, 32, 32, 4096, 4096, 128, 128, True),
    ("mla-2x16x8k", 2, 16, 16, 8192, 8192, 192, 128, False),
    ("mla-causal-1x128x8k", 1, 128, 128, 8192, 8192, 192, 128, True),
    ("mla-causal-4x16x4k", 4, 16, 16, 4096, 4096, 192, 128, True),
]


def aiter_kernels(q, k, v, o, causal, scale):
    """{name: (timed fn, fn that writes o)} for each AITER kernel that accepts this shape."""
    Sq, Skv, Dqk, Dv = q.shape[1], k.shape[1], q.shape[3], v.shape[3]
    out = {}

    if not causal and (Dqk, Dv) == (128, 128):

        def f_v4():
            mha_v4(q, k, v, BF16, BF16, BF16, softmax_scale=scale, out=o)

        out["v4"] = (f_v4, f_v4)

    def via_flash_attn(env):
        def f():
            for name in OPUS_ENV:
                os.environ.pop(name, None)
            os.environ.update(env)
            return flash_attn_func(q, k, v, softmax_scale=scale, causal=causal)

        return f, lambda: o.copy_(f())

    # flash_attn_func routes Sq <= 128 to CK rather than v3; every shape here has Sq > 128.
    if Sq > 128 and (Dqk, Dv) in ((128, 128), (192, 128)):
        out["v3"] = via_flash_attn({"AITER_DISABLE_FMHA_OPUS": "1"})
    if (Dqk, Dv) == (192, 128):
        out["opus"] = via_flash_attn({})
    elif (Dqk, Dv) == (128, 128) and Skv * k.stride(1) * k.element_size() < 1 << 32:
        # AITER falls back to v3 past a 4 GiB K extent; only time OPUS where it really runs.
        out["opus"] = via_flash_attn({"AITER_ENABLE_FMHA_OPUS": "1"})
    return out


@torch.no_grad()
def reference(q, k, v, causal, scale, b, heads, rows):
    """fp32 attention for batch b, the given q heads and q rows. BSHD inputs."""
    Hq, Hkv = q.shape[2], k.shape[2]
    Sq, Skv = q.shape[1], k.shape[1]
    out = {}
    for h in heads:
        kh = h // (Hq // Hkv)
        qq = q[b, rows, h].float()
        kk = k[b, :, kh].float()
        vv = v[b, :, kh].float()
        s = (qq @ kk.T) * scale
        if causal:
            ri = torch.as_tensor(rows, device=q.device).unsqueeze(1)
            cj = torch.arange(Skv, device=q.device).unsqueeze(0)
            s = s.masked_fill(cj > ri + (Skv - Sq), float("-inf"))
        out[h] = torch.softmax(s, -1) @ vv
    return out


def check(o, ref, b, rows):
    """Max abs error relative to the largest reference value; ok within 4 bf16 ulp of it (2^-6)."""
    err = peak = 0.0
    finite = True
    for h, r in ref.items():
        got = o[b, rows, h].float()
        err = max(err, (got - r).abs().max().item())
        peak = max(peak, r.abs().max().item())
        finite = finite and bool(torch.isfinite(got).all())
    rel = err / peak
    return {"max_abs": err, "max_rel": rel, "ok": finite and rel < 2**-6}


def time_graph(fn, n):
    """Mean ms per call over a HIP graph of n back-to-back calls (no host overhead)."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n):
            fn()
    g.replay()  # warm
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    g.replay()
    e1.record()
    e1.synchronize()
    del g
    return e0.elapsed_time(e1) / n


def time_eager(fn, n):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(n):
        fn()
    e1.record()
    e1.synchronize()
    return e0.elapsed_time(e1) / n


def make_inputs(B, Hq, Hkv, Sq, Skv, Dqk, Dv, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    q = torch.randn(B, Sq, Hq, Dqk, device=DEV, dtype=torch.bfloat16, generator=g)
    k = torch.randn(B, Skv, Hkv, Dqk, device=DEV, dtype=torch.bfloat16, generator=g)
    v = torch.randn(B, Skv, Hkv, Dv, device=DEV, dtype=torch.bfloat16, generator=g)
    return q, k, v


def run_shape(shape, passes, target_ms):
    label, B, Hq, Hkv, Sq, Skv, Dqk, Dv, causal = shape
    scale = Dqk**-0.5
    q, k, v = make_inputs(B, Hq, Hkv, Sq, Skv, Dqk, Dv)
    o_mm = torch.empty(B, Sq, Hq, Dv, device=DEV, dtype=torch.bfloat16)
    o_ai = torch.empty_like(o_mm)

    def f_mm():
        ma.forward(q, k, v, out=o_mm, layout="bshd", causal=causal, softmax_scale=scale)

    kernels = {"mm": (f_mm, f_mm), **aiter_kernels(q, k, v, o_ai, causal, scale)}
    res = {
        "label": label,
        "B": B,
        "Hq": Hq,
        "Hkv": Hkv,
        "Sq": Sq,
        "Skv": Skv,
        "Dqk": Dqk,
        "Dv": Dv,
        "causal": causal,
        "kernels": {},
    }

    # --- correctness: the same random rows and heads of every kernel's output against fp32
    gen = torch.Generator().manual_seed(1)
    rows = sorted(
        set(torch.randint(0, Sq, (min(Sq, 512),), generator=gen).tolist()) | {0, Sq - 1}
    )
    heads = sorted({0, Hq - 1, Hq // 2})
    bsel = B - 1
    ref = reference(q, k, v, causal, scale, bsel, heads, rows)
    for name, (_, write) in kernels.items():
        write()
        torch.cuda.synchronize()
        res["kernels"][name] = {
            "acc": check(o_mm if name == "mm" else o_ai, ref, bsel, rows)
        }
    del ref

    # --- iteration count so each graph is ~target_ms long
    t1 = time_eager(f_mm, 2)
    n = max(3, min(200, int(target_ms / max(t1, 1e-3))))
    res["iters"] = n

    names = list(kernels)
    hot = {k: [] for k in names}
    eager = {k: [] for k in names}
    for p in range(passes):
        for name in names[p % len(names) :] + names[: p % len(names)]:
            fn = kernels[name][0]
            try:
                hot[name].append(time_graph(fn, n))
            except Exception as e:  # graph capture unsupported: record and fall back
                res["kernels"][name]["graph_error"] = repr(e)[:200]
                hot[name].append(time_eager(fn, n))
            eager[name].append(time_eager(fn, n))

    for name in names:
        r = res["kernels"][name]
        for kind, d in (("hot", hot), ("eager", eager)):
            r[f"{kind}_ms"] = statistics.median(d[name])
        r["hot_spread"] = (max(hot[name]) - min(hot[name])) / r["hot_ms"]
    aiter = [name for name in names if name != "mm"]
    mm = res["kernels"]["mm"]
    for kind in ("hot", "eager"):
        best = min(aiter, key=lambda name: res["kernels"][name][f"{kind}_ms"])
        res[f"best_aiter_{kind}"] = best
        res[f"speedup_{kind}"] = res["kernels"][best][f"{kind}_ms"] / mm[f"{kind}_ms"]
    del q, k, v, o_mm, o_ai
    torch.cuda.empty_cache()
    return res


def summary(r):
    ks = r["kernels"]
    parts = [f"{r['label']:22s}"]
    for name, k in ks.items():
        bad = "" if k["acc"]["ok"] else "!!"
        parts.append(
            f"{name} {k['hot_ms']:.4f} ms "
            f"(±{k['hot_spread']:.1%}, err {k['acc']['max_rel']:.1e}{bad})"
        )
    parts.append(
        f"| x{r['speedup_hot']:.3f} vs {r['best_aiter_hot']}, x{r['speedup_eager']:.3f} eager"
    )
    return "  ".join(parts)


def markdown(results):
    """README table: ms against the fastest AITER kernel; speedup = aiter_ms / ours_ms."""
    lines = [
        "| B, Hq/Hkv, Sq/Skv, Dqk/Dv | Causal | Ours (ms) | Fastest AITER (ms) | Speedup "
        "| Rel. error, ours / AITER |",
        "|---|---|---|---|---|---|",
    ]
    ok = [r for r in results if "error" not in r]
    for r in ok:
        heads = f"{r['Hq']}" if r["Hq"] == r["Hkv"] else f"{r['Hq']}/{r['Hkv']}"
        seq = f"{r['Sq']}" if r["Sq"] == r["Skv"] else f"{r['Sq']}/{r['Skv']}"
        dims = f"{r['Dqk']}" if r["Dqk"] == r["Dv"] else f"{r['Dqk']}/{r['Dv']}"
        mm, best = r["kernels"]["mm"], r["best_aiter_hot"]
        ai = r["kernels"][best]
        a, b = mm["hot_ms"], ai["hot_ms"]
        flag = (
            ""
            if all(k["acc"]["ok"] for k in r["kernels"].values())
            else " (accuracy check failed)"
        )
        lines.append(
            f"| {r['B']}, {heads}, {seq}, {dims}{flag} | {'yes' if r['causal'] else 'no'} "
            f"| {'**' if a < b else ''}{a:#.4g}{'**' if a < b else ''} "
            f"| {'**' if b < a else ''}{b:#.4g}{'**' if b < a else ''} ({best}) "
            f"| {r['speedup_hot']:.2f}× "
            f"| {mm['acc']['max_rel']:.1e} / {ai['acc']['max_rel']:.1e} |"
        )
    if ok:
        gm = math.exp(statistics.mean(math.log(r["speedup_hot"]) for r in ok))
        lines.append(f"| _geomean_ | | | | _{gm:.2f}×_ | |")
    for r in results:
        if "error" in r:
            lines.append(f"\n{r['label']} failed: {r['error']}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--passes", type=int, default=7)
    ap.add_argument("--target-ms", type=float, default=200.0)
    ap.add_argument("--only", default=None, help="substring filter on labels")
    ap.add_argument("--json", default=None, help="also write every result to this file")
    ap.add_argument(
        "--markdown", action="store_true", help="print the README table at the end"
    )
    a = ap.parse_args()

    print(
        torch.cuda.get_device_name(0), torch.__version__, torch.version.hip, flush=True
    )
    shapes = [s for s in SHAPES if not a.only or a.only in s[0]]
    out = []
    for shape in shapes:
        try:
            r = run_shape(shape, a.passes, a.target_ms)
            print(summary(r), flush=True)
        except Exception as e:
            r = {"label": shape[0], "error": repr(e)[:400]}
            print("ERROR", shape[0], r["error"], flush=True)
            torch.cuda.empty_cache()
        out.append(r)
        if a.json:
            with open(a.json, "w") as f:
                json.dump(out, f, indent=1)
    if a.markdown:
        print()
        print(markdown(out))


if __name__ == "__main__":
    main()
