"""Array backend selection: CuPy on GPU when available, otherwise NumPy.

``_np`` is bound once at import time and never rebound, so other modules may
safely import it directly (``from worldsim.core.backend import _np``).
"""

from __future__ import annotations

try:  # pragma: no cover - optional dependency
    import cupy as _cupy  # GPU-backed drop-in replacement for numpy
    _cupy.zeros(1)  # verify driver availability
    _np = _cupy
    USING_GPU = True
except Exception:  # pragma: no cover - fallback path
    try:
        import numpy as _np
    except Exception:  # pragma: no cover - last resort
        _np = None
    USING_GPU = False


def get_array_module(require: bool = False):
    """Return the configured array module or ``None``.

    The project prefers :mod:`cupy` when available and will otherwise fall
    back to :mod:`numpy`.  When *require* is ``True`` a
    :class:`RuntimeError` is raised if neither backend can be imported.
    """

    if _np is None:
        if require:
            raise RuntimeError("numpy or cupy is required but not installed")
        return None
    return _np


def as_array(data, *, dtype=None):
    """Convert *data* into an array using the active backend.

    When no accelerated backend is available this function raises
    :class:`RuntimeError`.  The helper mirrors :func:`numpy.asarray`/``cupy``'s
    semantics so higher level modules can transparently work with GPU backed
    arrays when present.
    """

    xp = get_array_module(require=True)
    return xp.asarray(data, dtype=dtype)
