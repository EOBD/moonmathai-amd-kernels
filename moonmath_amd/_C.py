"""moonmath_amd._C: the gfx942 (CDNA3) module under its name from before the per-arch split, so code that
imports moonmath_amd._C keeps working. New code: moonmath_amd._arch.load_module("gfx942")."""

from moonmath_amd._C_gfx942 import *  # noqa: F401,F403
