"""Utility helpers for WorldSim.

The module now tries to import ``cupy`` to accelerate array operations on the
GPU. If unavailable it falls back to ``numpy`` or pure Python. The helper
functions operate on scalars or arrays transparently.
"""

from __future__ import annotations

import asyncio
import subprocess
import shutil
from pathlib import Path
from typing import Iterable, Awaitable
import ctypes
import os
import multiprocessing as _mp
from contextlib import contextmanager
import signal

_lib = None
_lib_path = Path(__file__).resolve().parent / "cpp_utils" / "libcpp_utils.so"
_rust_lib = None
_rust_lib_path = Path(__file__).resolve().parent / "rust_utils" / "librust_utils.so"


def _rust_available() -> bool:
    return shutil.which("cargo") is not None


USE_RUST = True
env_flag = os.getenv("WORLDSIM_USE_RUST", "auto").lower()
if env_flag in {"0", "false", "no"}:
    USE_RUST = False
elif env_flag in {"1", "true", "yes"}:
    USE_RUST = _rust_available()
else:  # auto
    USE_RUST = _rust_available()


def _build_cpp_utils() -> None:
    """Compile optional C++ helpers if missing."""
    if _mp.current_process().name != "MainProcess" or _lib_path.exists():
        return
    cpp_file = _lib_path.with_name("utils.cpp")
    if not cpp_file.exists():
        return
    cmd = ["g++", "-O2", "-fPIC", "-shared", str(cpp_file), "-o", str(_lib_path)]
    try:
        subprocess.run(cmd, check=False)
    except Exception:
        return


def _build_rust_utils() -> None:
    """Compile optional Rust helpers if missing."""
    if not USE_RUST or _mp.current_process().name != "MainProcess" or _rust_lib_path.exists():
        return
    cargo = shutil.which("cargo")
    if cargo is None:
        return
    src_dir = _rust_lib_path.parent
    try:
        subprocess.run([cargo, "build", "--release"], cwd=src_dir, check=False)
    except Exception:
        return
    built = src_dir / "target" / "release" / "librust_utils.so"
    if built.exists():
        try:
            shutil.copy(built, _rust_lib_path)
        except Exception:
            return


_build_cpp_utils()
if _lib_path.exists():
    try:
        _lib = ctypes.CDLL(str(_lib_path))
        _lib.cpp_logistic_growth.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int]
        _lib.cpp_logistic_growth.restype = ctypes.c_double
        _lib.cpp_distance.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int]
        _lib.cpp_distance.restype = ctypes.c_double
    except OSError:
        _lib = None

if USE_RUST:
    _build_rust_utils()
if USE_RUST and _rust_lib_path.exists():
    try:
        _rust_lib = ctypes.CDLL(str(_rust_lib_path))
        _rust_lib.ru_logistic_growth.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int]
        _rust_lib.ru_logistic_growth.restype = ctypes.c_double
        _rust_lib.ru_distance.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int]
        _rust_lib.ru_distance.restype = ctypes.c_double
        _rust_lib.ru_polygon_area.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int]
        _rust_lib.ru_polygon_area.restype = ctypes.c_double
        _rust_lib.ru_polygon_centroid.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
        _rust_lib.ru_polygon_centroid.restype = None
        _rust_lib.ru_logistic_growth_batch.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
        ]
        _rust_lib.ru_logistic_growth_batch.restype = None
    except OSError:
        _rust_lib = None


def set_use_rust(flag: bool) -> None:
    """Enable or disable Rust helpers at runtime."""
    global USE_RUST, _rust_lib
    USE_RUST = flag and _rust_available()
    if USE_RUST:
        _build_rust_utils()
        if _rust_lib_path.exists():
            try:
                _rust_lib = ctypes.CDLL(str(_rust_lib_path))
            except OSError:
                _rust_lib = None
    else:
        _rust_lib = None

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

# Global toggle controlling verbosity of print statements
VERBOSE: bool = True

# Global toggle for approximate calculations
APPROXIMATE: bool = False

# Global toggle for profiling output
DEBUG_PROFILE: bool = False

# When set to a nation name string, wprint() only emits output that is
# attributed to that nation (all other nations are silenced).  Set to None
# to show output for every nation.
WATCHED_NATION: str | None = None

# Average speeds for logistics (kilometres per day)
# Surface travel assumes a baseline of ``365/5`` days to cover 365 km,
# i.e. roughly 73 km travelled per day at ~3 km/h.
LAND_SPEED_KM_PER_DAY: float = 365 / 5
# Space travel is assumed to be much faster – about a thousand times
# quicker than surface movement.
SPACE_SPEED_KM_PER_DAY: float = LAND_SPEED_KM_PER_DAY * 1000


def vprint(*args, **kwargs) -> None:
    """Print only when ``VERBOSE`` is ``True``."""
    if VERBOSE:
        print(*args, **kwargs)


def wprint(nation_name: str, *args, **kwargs) -> None:
    """Print only when ``VERBOSE`` and the nation is being watched.

    When :data:`WATCHED_NATION` is ``None`` all nations are shown.  When it
    is set to a name string only messages attributed to that nation appear.
    This avoids building and printing strings for every nation in the
    simulation when the user cares about just one.
    """
    if VERBOSE and (WATCHED_NATION is None or WATCHED_NATION == nation_name):
        print(*args, **kwargs)


def _compact_pop(n: int) -> str:
    """Format a population count as a short human-readable string.

    Examples: ``1234567`` → ``"1.2M"``, ``45000`` → ``"45k"``, ``999`` → ``"999"``.
    """
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n // 1_000}k"
    return str(n)


async def gather_batches(
    coroutines: Iterable[Awaitable], batch_size: int = 8
) -> None:
    """Run *coroutines* in batches of ``batch_size``."""

    tasks = list(coroutines)
    for i in range(0, len(tasks), batch_size):
        await asyncio.gather(*tasks[i : i + batch_size])


@contextmanager
def time_limit(seconds: float | None):
    """Raise :class:`TimeoutError` if the block exceeds ``seconds``.

    On platforms without ``SIGALRM`` (e.g. Windows) or when ``seconds`` is
    ``None``/non-positive, this behaves as a no-op.
    """

    if seconds is None or seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def handler(signum, frame):  # pragma: no cover - platform specific
        raise TimeoutError("operation timed out")

    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


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

    if _rust_lib is not None and not hasattr(n, "__len__"):
        return float(
            _rust_lib.ru_logistic_growth(float(n), float(r), float(k), int(APPROXIMATE))
        )
    if _lib is not None and not hasattr(n, "__len__"):
        return float(_lib.cpp_logistic_growth(float(n), float(r), float(k), int(APPROXIMATE)))
    if not hasattr(n, "__len__"):
        return n + r * n * (1 - n / k) if not APPROXIMATE else n + r * n
    if hasattr(n, "__len__") and _np is not None:
        return n + r * n * (1 - n / k)
    if hasattr(n, "__len__") and _rust_lib is not None:
        arr_n = (ctypes.c_double * len(n))(*[float(x) for x in n])
        if hasattr(k, "__len__"):
            arr_k = (ctypes.c_double * len(k))(*[float(x) for x in k])
        else:
            arr_k = (ctypes.c_double * len(n))(*[float(k) for _ in n])
        out = (ctypes.c_double * len(n))()
        _rust_lib.ru_logistic_growth_batch(
            arr_n,
            arr_k,
            len(n),
            float(r),
            int(APPROXIMATE),
            out,
        )
        return [out[i] for i in range(len(n))]
    if hasattr(n, "__len__"):
        if APPROXIMATE:
            return [val + r * val for val in n]
    if APPROXIMATE:
        return n + r * n
    if _np is not None:
        return n + r * n * (1 - n / k)
    return n + r * n * (1 - n / k)


def distance(p1, p2):
    """Euclidean distance between two points."""
    if len(p1) != len(p2):
        raise ValueError("Points must have same dimension")
    if _rust_lib is not None:
        arr1 = (ctypes.c_double * len(p1))(*[float(x) for x in p1])
        arr2 = (ctypes.c_double * len(p2))(*[float(x) for x in p2])
        return float(_rust_lib.ru_distance(arr1, arr2, len(p1)))
    if _lib is not None:
        arr1 = (ctypes.c_double * len(p1))(*[float(x) for x in p1])
        arr2 = (ctypes.c_double * len(p2))(*[float(x) for x in p2])
        return float(_lib.cpp_distance(arr1, arr2, len(p1)))
    if _np is not None:
        arr1 = _np.array(p1)
        arr2 = _np.array(p2)
        return float(_np.sqrt(((arr1 - arr2) ** 2).sum()))
    return sum((a - b) ** 2 for a, b in zip(p1, p2)) ** 0.5


def travel_time(distance_km: float, *, space: bool = False) -> float:
    """Return travel time in days for ``distance_km``.

    Distances are rounded to the nearest metre (three decimals) to match
    the project's precision requirements. The baseline speed is derived
    from ``365/5`` – roughly 73 km covered per day. When ``space`` is
    ``True`` a much higher velocity is assumed for interplanetary travel.
    """

    dist = round(distance_km, 3)
    speed = SPACE_SPEED_KM_PER_DAY if space else LAND_SPEED_KM_PER_DAY
    return round(dist / speed, 3)


def distance_sq(p1, p2):
    """Squared Euclidean distance between two points."""
    n = len(p1)
    if n != len(p2):
        raise ValueError("Points must have same dimension")
    # Fast scalar paths for the 2-D and 3-D cases used throughout the codebase;
    # avoids the overhead of allocating numpy arrays for tiny inputs.
    if n == 2:
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return dx * dx + dy * dy
    if n == 3:
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        dz = p1[2] - p2[2]
        return dx * dx + dy * dy + dz * dz
    if _np is not None:
        arr1 = _np.array(p1)
        arr2 = _np.array(p2)
        diff = arr1 - arr2
        return float(_np.dot(diff, diff))
    return sum((a - b) * (a - b) for a, b in zip(p1, p2))


def polygon_area(coords):
    if len(coords) < 3:
        return 0.0
    if APPROXIMATE:
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        return (max(xs) - min(xs)) * (max(ys) - min(ys))
    if _rust_lib is not None:
        flat = [coord for xy in coords for coord in xy]
        arr = (ctypes.c_double * len(flat))(*[float(v) for v in flat])
        return float(_rust_lib.ru_polygon_area(arr, len(coords)))
    if _np is not None:
        arr = _np.asarray(coords)
        x1 = arr[:, 0]
        y1 = arr[:, 1]
        x2 = _np.roll(x1, -1)
        y2 = _np.roll(y1, -1)
        area = _np.sum(x1 * y2 - x2 * y1)
        return float(_np.abs(area) / 2.0)
    area = 0.0
    for i, (x1, y1) in enumerate(coords):
        x2, y2 = coords[(i + 1) % len(coords)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def polygon_centroid(coords):
    if not coords:
        return (0.0, 0.0)
    if APPROXIMATE:
        x = sum(c[0] for c in coords) / len(coords)
        y = sum(c[1] for c in coords) / len(coords)
        return (x, y)
    area = polygon_area(coords)
    if area == 0:
        x = sum(c[0] for c in coords) / len(coords)
        y = sum(c[1] for c in coords) / len(coords)
        return (x, y)
    cx = 0.0
    cy = 0.0
    factor = 0.0
    if _rust_lib is not None:
        flat = [coord for xy in coords for coord in xy]
        arr = (ctypes.c_double * len(flat))(*[float(v) for v in flat])
        out_x = ctypes.c_double()
        out_y = ctypes.c_double()
        _rust_lib.ru_polygon_centroid(arr, len(coords), ctypes.byref(out_x), ctypes.byref(out_y))
        return (out_x.value, out_y.value)
    if _np is not None:
        arr = _np.asarray(coords)
        x1 = arr[:, 0]
        y1 = arr[:, 1]
        x2 = _np.roll(x1, -1)
        y2 = _np.roll(y1, -1)
        cross = x1 * y2 - x2 * y1
        factor = _np.sum(cross)
        cx = _np.sum((x1 + x2) * cross)
        cy = _np.sum((y1 + y2) * cross)
        factor *= 3
        if factor == 0:
            return (0.0, 0.0)
        return (float(cx / factor), float(cy / factor))
    for i, (x1, y1) in enumerate(coords):
        x2, y2 = coords[(i + 1) % len(coords)]
        cross = x1 * y2 - x2 * y1
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
        factor += cross
    factor *= 3
    if factor == 0:
        return (0.0, 0.0)
    return (cx / factor, cy / factor)
