"""Array-backend selection for the Nelder-Mead AI stack.

By default the controllers run on the project-wide backend (CuPy when a GPU
is available, otherwise NumPy, otherwise pure Python).  The choice can be
overridden with the ``WORLDSIM_AI_BACKEND`` environment variable:

``auto``   (default) CuPy → NumPy → pure Python
``cupy``   force CuPy, fall back to NumPy when unavailable
``numpy``  force NumPy even when CuPy is installed
``python`` disable array maths entirely (pure-Python lists)
"""

from __future__ import annotations

import os

from ..core.backend import get_array_module


def ai_array_module():
    """Return the array module the AI controllers should use (or ``None``)."""
    choice = os.getenv("WORLDSIM_AI_BACKEND", "auto").lower()
    if choice == "python":
        return None
    if choice == "numpy":
        try:
            import numpy
            return numpy
        except ImportError:  # pragma: no cover - numpy is a requirement
            return None
    if choice == "cupy":
        xp = get_array_module()
        if xp is not None and getattr(xp, "__name__", "") == "cupy":
            return xp
        try:
            import numpy
            return numpy
        except ImportError:  # pragma: no cover
            return None
    return get_array_module()
