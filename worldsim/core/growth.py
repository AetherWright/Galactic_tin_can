"""Logistic growth — the demographic workhorse of the simulation.

Scalars dispatch to Rust, then C++, then Python; arrays use the active
NumPy/CuPy backend or a Rust batch kernel.
"""

from __future__ import annotations

import ctypes

from . import flags, native
from .backend import _np


def logistic_growth(n: float | "_np.ndarray", r: float, k: float) -> float | "_np.ndarray":
    """Return the next step in a logistic growth sequence.

    The computation runs on ``cupy`` if available, otherwise ``numpy`` or pure
    Python. Arrays are supported transparently. ``k`` values less than or equal
    to zero fall back to exponential growth to avoid divide by zero errors.
    """

    if (not hasattr(k, "__len__") and k <= 0) or (
        hasattr(k, "__len__") and any(cap <= 0 for cap in k)
    ):
        if hasattr(n, "__len__"):
            if not hasattr(k, "__len__"):
                caps = [k for _ in n]
            else:
                caps = list(k)
            result = []
            for val, cap in zip(n, caps):
                if cap <= 0:
                    result.append(val + r * val)
                else:
                    result.append(val + r * val * (1 - val / cap))
            return _np.array(result) if _np is not None else result
        return n + r * n

    if native._rust_lib is not None and not hasattr(n, "__len__"):
        return float(
            native._rust_lib.ru_logistic_growth(float(n), float(r), float(k), int(flags.APPROXIMATE))
        )
    if native._lib is not None and not hasattr(n, "__len__"):
        return float(native._lib.cpp_logistic_growth(float(n), float(r), float(k), int(flags.APPROXIMATE)))
    if not hasattr(n, "__len__"):
        return n + r * n * (1 - n / k) if not flags.APPROXIMATE else n + r * n
    if hasattr(n, "__len__") and _np is not None:
        return n + r * n * (1 - n / k)
    if hasattr(n, "__len__") and native._rust_lib is not None:
        arr_n = (ctypes.c_double * len(n))(*[float(x) for x in n])
        if hasattr(k, "__len__"):
            arr_k = (ctypes.c_double * len(k))(*[float(x) for x in k])
        else:
            arr_k = (ctypes.c_double * len(n))(*[float(k) for _ in n])
        out = (ctypes.c_double * len(n))()
        native._rust_lib.ru_logistic_growth_batch(
            arr_n,
            arr_k,
            len(n),
            float(r),
            int(flags.APPROXIMATE),
            out,
        )
        return [out[i] for i in range(len(n))]
    if hasattr(n, "__len__"):
        if flags.APPROXIMATE:
            return [val + r * val for val in n]
    if flags.APPROXIMATE:
        return n + r * n
    if _np is not None:
        return n + r * n * (1 - n / k)
    return n + r * n * (1 - n / k)
