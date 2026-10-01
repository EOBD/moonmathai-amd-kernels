"""The live GPU's architecture, and the extension module built for it.

setup.py builds one module per architecture folder in csrc/, each only for its own arch (GPU_ARCHS):
moonmath_amd._C_gfx942 (CDNA3, MI300X). An op picks its module at call time from the device it runs
on, as AITER's get_gfx_runtime() does.
"""

import functools
import importlib
import importlib.util

import torch

ARCHS = ("gfx942",)


@functools.lru_cache(maxsize=None)
def _arch_of(index: int) -> str:
    return torch.cuda.get_device_properties(index).gcnArchName.split(":")[0]


def get_gfx_runtime(device=None) -> str:
    """The gfx arch of `device` (default: the current GPU), e.g. "gfx942", without target features."""
    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError("moonmath_amd: no ROCm GPU is visible")
        return _arch_of(torch.cuda.current_device())
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError(f"moonmath_amd: expected a GPU device, got {device}")
    return _arch_of(
        torch.cuda.current_device() if device.index is None else device.index
    )


def is_built(arch: str) -> bool:
    """Whether moonmath_amd._C_<arch> was built."""
    return importlib.util.find_spec(f"moonmath_amd._C_{arch}") is not None


@functools.lru_cache(maxsize=None)
def load_module(arch: str):
    """moonmath_amd._C_<arch>, or an ImportError that says how to build it."""
    name = f"moonmath_amd._C_{arch}"
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as e:
        if e.name != name:
            raise
        raise ImportError(
            f"{name} is not built: this moonmath_amd was built without {arch} kernels. Rebuild with "
            f'GPU_ARCHS={arch} (or a list that includes it, e.g. GPU_ARCHS="gfx942"): '
            "pip install -e . --no-build-isolation"
        ) from None


@functools.lru_cache(maxsize=1)
def _only_gfx942() -> bool:
    """Whether every visible GPU is gfx942: then the per-call check is free, as before the arch split."""
    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    return count > 0 and all(_arch_of(i) == "gfx942" for i in range(count))


def require_gfx942(name: str, device=None):
    """A clear error when `device` (default: the current GPU) is not gfx942, for CDNA3-only ops."""
    if _only_gfx942():
        return
    if torch.cuda.is_available():
        arch = get_gfx_runtime(device)
        if arch != "gfx942":
            raise RuntimeError(
                f"moonmath_amd.{name} runs on gfx942 (CDNA3, MI300X) only; the current GPU is {arch}"
            )


def gfx942_op(name: str):
    """A gfx942-only op of the CDNA3 module; a clear error when the current GPU is another arch."""
    require_gfx942(name)
    return getattr(load_module("gfx942"), name)
