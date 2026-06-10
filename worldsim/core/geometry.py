"""Distance, travel-time and polygon helpers.

Scalar hot-paths dispatch to the Rust/C++ accelerators when loaded, then to
the array backend, then to pure Python.  Native library handles are read
through :mod:`worldsim.core.native` on every call because
:func:`worldsim.core.native.set_use_rust` can swap them at runtime.
"""

from __future__ import annotations

import ctypes

from . import flags, native
from .backend import _np

# Average speeds for logistics (kilometres per day)
# Surface travel assumes a baseline of ``365/5`` days to cover 365 km,
# i.e. roughly 73 km travelled per day at ~3 km/h.
LAND_SPEED_KM_PER_DAY: float = 365 / 5
# Space travel is assumed to be much faster – about a thousand times
# quicker than surface movement.
SPACE_SPEED_KM_PER_DAY: float = LAND_SPEED_KM_PER_DAY * 1000


def distance(p1, p2):
    """Euclidean distance between two points."""
    if len(p1) != len(p2):
        raise ValueError("Points must have same dimension")
    if native._rust_lib is not None:
        arr1 = (ctypes.c_double * len(p1))(*[float(x) for x in p1])
        arr2 = (ctypes.c_double * len(p2))(*[float(x) for x in p2])
        return float(native._rust_lib.ru_distance(arr1, arr2, len(p1)))
    if native._lib is not None:
        arr1 = (ctypes.c_double * len(p1))(*[float(x) for x in p1])
        arr2 = (ctypes.c_double * len(p2))(*[float(x) for x in p2])
        return float(native._lib.cpp_distance(arr1, arr2, len(p1)))
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
    if flags.APPROXIMATE:
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        return (max(xs) - min(xs)) * (max(ys) - min(ys))
    if native._rust_lib is not None:
        flat = [coord for xy in coords for coord in xy]
        arr = (ctypes.c_double * len(flat))(*[float(v) for v in flat])
        return float(native._rust_lib.ru_polygon_area(arr, len(coords)))
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
    if flags.APPROXIMATE:
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
    if native._rust_lib is not None:
        flat = [coord for xy in coords for coord in xy]
        arr = (ctypes.c_double * len(flat))(*[float(v) for v in flat])
        out_x = ctypes.c_double()
        out_y = ctypes.c_double()
        native._rust_lib.ru_polygon_centroid(arr, len(coords), ctypes.byref(out_x), ctypes.byref(out_y))
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
