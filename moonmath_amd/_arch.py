"""Architecture of the live GPU and the matching extension module.

setup.py builds one module per csrc/ architecture folder, each for its own arch only (GPU_ARCHS):
moonmath_amd._C_gfx942 (CDNA3, MI300X) and moonmath_amd._C_gfx950 (CDNA4, MI355X). At call time an
op selects its module from the device it runs on, mirroring AITER's get_gfx_runtime().
"""

import functools
import importlib
import importlib.util

import torch

ARCHS = ("gfx942", "gfx950")


@functools.lru_cache(maxsize=None)
def _arch_of(index: int) -> str:
    return torch.cuda.get_device_properties(index).gcnArchName.split(":")[0]


def get_gfx_runtime(device=None) -> str:
    """gfx arch of `device` (defaults to the current GPU) without target features, e.g. "gfx950"."""
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
    """True if moonmath_amd._C_<arch> is built."""
    return importlib.util.find_spec(f"moonmath_amd._C_{arch}") is not None


@functools.lru_cache(maxsize=None)
def load_module(arch: str):
    """Return moonmath_amd._C_<arch>, or raise an ImportError explaining how to build it."""
    name = f"moonmath_amd._C_{arch}"
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as e:
        if e.name != name:
            raise
        raise ImportError(
            f"{name} is not built: this moonmath_amd was built without {arch} kernels. Rebuild with "
            f'GPU_ARCHS={arch} (or a list that includes it, e.g. GPU_ARCHS="gfx942;gfx950"): '
            "pip install -e . --no-build-isolation"
        ) from None


@functools.lru_cache(maxsize=1)
def _only_gfx942() -> bool:
    """True when all visible GPUs are gfx942: the per-call check then costs nothing, as before the arch split."""
    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    return count > 0 and all(_arch_of(i) == "gfx942" for i in range(count))


def require_gfx942(name: str, device=None):
    """Raise a clear error when `device` (default: the current GPU) is not gfx942; for CDNA3-only ops."""
    if _only_gfx942():
        return
    if torch.cuda.is_available():
        arch = get_gfx_runtime(device)
        if arch != "gfx942":
            raise RuntimeError(
                f"moonmath_amd.{name} runs on gfx942 (CDNA3, MI300X) only; the current GPU is {arch}"
            )


def gfx942_op(name: str):
    """Fetch a gfx942-only op from the CDNA3 module, erroring clearly on any other arch."""
    require_gfx942(name)
    return getattr(load_module("gfx942"), name)
