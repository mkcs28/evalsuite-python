"""Access to the compiled core (``evalsuite._core``, built from rust/).

Every function that has a compiled kernel keeps its Python implementation: it is the reference the kernel is
tested against and the fallback when the extension is missing (the pure-Python wheel, an unsupported platform)
or when ``EVALSUITE_PURE_PYTHON=1`` is set. Both give identical results.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Optional

core: Optional[Any]
if os.environ.get("EVALSUITE_PURE_PYTHON", "").strip().lower() in ("1", "true", "yes", "on"):
    core = None
else:
    try:
        from . import _core as core  # type: ignore[attr-defined, no-redef]
    except ImportError:  # pure-Python wheel or source checkout without a build
        core = None

HAVE_NATIVE = core is not None

# Python >= 3.12 sums floats with compensated (Neumaier) summation in the built-in sum(); kernels that port code
# using sum() over floats take this flag so their results match the running interpreter bit for bit.
PY_SUM_COMPENSATED = sys.version_info >= (3, 12)


def backend() -> str:
    """``"rust"`` when the compiled core is in use, ``"python"`` otherwise."""
    return "rust" if HAVE_NATIVE else "python"


def array_view(arr: Any, dtype: Any) -> tuple[Any, tuple[int, int]]:
    """A C-contiguous array of exactly ``dtype`` and its ``(data address, element count)`` view for the core.

    The returned array must stay referenced (keep it in a local variable) until the core call returns; the
    core only reads from it."""
    import numpy as np

    a = np.ascontiguousarray(arr, dtype=dtype)
    if not a.flags.c_contiguous or a.dtype != np.dtype(dtype):  # pragma: no cover - guaranteed by NumPy
        raise ValueError("array must be C-contiguous with the requested dtype")
    return a, (int(a.ctypes.data), int(a.size))


class Vocab:
    """Maps tokens to consecutive integer ids (0, 1, 2, …) so token sequences can cross into the compiled core."""

    __slots__ = ("ids",)

    def __init__(self) -> None:
        self.ids: dict[Any, int] = {}

    def encode(self, tokens: Any) -> list[int]:
        ids = self.ids
        get = ids.get
        out = []
        for t in tokens:
            i = get(t)
            if i is None:
                i = ids[t] = len(ids)
            out.append(i)
        return out
