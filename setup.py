"""Build one torch CUDAExtension per GPU architecture folder in csrc/ (ROCm-only).

    csrc/gfx942/  CDNA3 (MI300X): MHA forward, MLA a16w8 decode, MXFP4 MoE  -> moonmath_amd._C_gfx942

Targets come from GPU_ARCHS, as in AITER: "native" (default; the GPUs rocminfo lists) or a list such as
"gfx942". A module is built when its arch is a target, and only for that arch. CDNA3_ARCH, the old
single-arch knob, is used when GPU_ARCHS is not set; with neither set and no GPU found, the build targets
gfx942, as it did before.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path

import torch
from setuptools import setup
from setuptools.command.build import build as _build
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Fail fast if someone tries to build against a non-ROCm torch
if torch.version.hip is None:
    raise RuntimeError(
        "moonmath_amd requires a ROCm build of PyTorch. "
        "torch.version.hip is None, indicating this is not a ROCm installation. "
        "Install ROCm PyTorch and ensure hipcc is available on PATH."
    )

ROOT = Path(__file__).parent.resolve()
CSRC = ROOT / "csrc"
DIST = ROOT / "dist"

_HIP_FLAGS = ["-O3", "-std=c++17", "-ffast-math", "-fno-math-errno", "-mllvm", "-amdgpu-early-inline-all=true"]

# One module per arch folder, listing its compile units (other files there are headers) and device flags.
MODULES = {
    "gfx942": dict(
        sources=[
            "attention_api.cpp",
            "attention_rtna.hip",
            "attention_rtne.hip",
            "attention_rtz.hip",
            # ── MLA (DeepSeek-V3): absorbed decode with bf16 Q and fp8 KV ──
            "mla_decode_a16w8_api.cpp",  # pybind entry for mla_decode_a16w8 and mla_dcp_lse_merge_ranks
            "mla_decode_a16w8.hip",      # a16w8 absorbed decode over a dense paged batch: any q_len, H 1..128,
                                         #   optional DCP: (position, head) rows in query tiles of 16..96 rows,
                                         #   run against a 16-token KV tile with a first-class base-2 LSE
            # ── MXFP4 MoE GEMMs ──
            "mxfp4_moe_api.cpp",         # pybind entry for mxfp4_moe_gateup/_down and their planners
            "mxfp4_moe_gateup.hip",      # gate/up projection with 16/32/48 tiles and EPI_NONE or SituGLU
            "mxfp4_moe_down.hip",        # down projection, sweeping an LDS-staged A tile over n chunks
        ],
        hip_flags=_HIP_FLAGS,
    ),
}


def _native_archs():
    """The gfx archs of the GPUs rocminfo lists, as AITER's "native"."""
    try:
        out = subprocess.run(["rocminfo"], capture_output=True, text=True, check=False).stdout
    except OSError as e:
        raise RuntimeError("GPU_ARCHS=native needs rocminfo; set GPU_ARCHS, e.g. GPU_ARCHS=gfx942") from e
    archs = sorted(set(re.findall(r"^\s*Name:\s+(gfx\w+)\s*$", out, re.MULTILINE)))
    if not archs:
        raise RuntimeError("rocminfo lists no GPU; set GPU_ARCHS, e.g. GPU_ARCHS=gfx942")
    return archs


def build_targets():
    """{arch folder: offload targets}. A target may carry features, e.g. gfx942:xnack-."""
    env = os.environ.get("GPU_ARCHS") or os.environ.get("CDNA3_ARCH")
    targets = [t.strip() for t in (env or "native").split(";") if t.strip()]
    if not targets:
        raise RuntimeError(f"GPU_ARCHS={env!r} names no arch; supported: {', '.join(MODULES)}")
    if targets == ["native"]:
        try:
            targets = _native_archs()
        except RuntimeError:
            if env:                         # GPU_ARCHS=native explicitly asked for this machine's GPUs
                raise
            targets = ["gfx942"]            # no knob and no GPU (e.g. a build container): keep the old default
            print("moonmath_amd: no GPU found and GPU_ARCHS not set: building for gfx942")
    by_arch = {}
    for t in targets:
        arch = t.split(":")[0]
        if arch not in MODULES:
            raise RuntimeError(f"GPU_ARCHS: moonmath_amd has no kernels for {t}; supported: {', '.join(MODULES)}")
        by_arch.setdefault(arch, []).append(t)
    return by_arch


# Copy csrc/<arch>/ into dist/<arch>/ and build from there: torch's ROCm hipify
# writes its generated `<stem>_hip.cpp` beside the source, so building from dist/
# keeps those intermediates out of csrc/.
def stage(arch):
    for src in (CSRC / arch).rglob("*"):
        dst = DIST / src.relative_to(CSRC)
        if src.is_file() and (not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime):
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    return [str((DIST / arch / name).relative_to(ROOT)) for name in MODULES[arch]["sources"]]


def extension(arch, targets):
    return CUDAExtension(
        name=f"moonmath_amd._C_{arch}",
        sources=stage(arch),
        # ROCm-only flags; BuildExtension routes "nvcc" to hipcc on ROCm, so no is_rocm branch is needed
        extra_compile_args={
            "cxx": ["-O3"],                  # torch contributes the -std its headers require
            "nvcc": MODULES[arch]["hip_flags"] + [f"--offload-arch={t}" for t in targets],
        },
    )


class DistBuild(_build):
    """Route every build artifact under dist/.

    Setting ``build_base`` makes ``build_ext`` (object files, ninja temp files,
    the assembled ``lib.*`` tree) and ``bdist`` (wheel staging) all live under
    dist/ instead of spawning a separate top-level build/ directory.
    """

    def initialize_options(self):
        super().initialize_options()
        self.build_base = str(DIST)


setup(
    ext_modules=[extension(arch, targets) for arch, targets in build_targets().items()],
    cmdclass={"build": DistBuild, "build_ext": BuildExtension},
)
