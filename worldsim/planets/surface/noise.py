"""Core noise primitives for procedural surface generation.

Runs on CuPy when a GPU is available, otherwise NumPy — the public helpers
always return CPU numpy arrays so callers never handle CuPy directly.
"""

from __future__ import annotations

import numpy as _np_cpu  # always available; used for fallback and final conversion
try:  # pragma: no cover - optional GPU backend
    import cupy as _xp
    _GPU_HEIGHTMAP = True
except Exception:  # pragma: no cover
    _xp = _np_cpu  # type: ignore[assignment]
    _GPU_HEIGHTMAP = False


def _to_numpy(arr) -> "_np_cpu.ndarray":
    """Return a CPU numpy array, calling ``.get()`` on CuPy arrays."""
    if _GPU_HEIGHTMAP and hasattr(arr, "get"):
        return arr.get()
    return arr

# Convenience alias so downstream modules read as plain numpy.
np = _np_cpu  # used only for type-stable constants and random generators


def _hash(xi, yi, seed: int):
    """Vectorised integer-coordinate hash returning values in [-0.5, 0.5).

    Uses a sequence of multiply-xor-shift operations (Murmur-style) that
    produce a well-distributed pseudo-random float from any (xi, yi) pair.
    Negative coordinates are handled correctly — the arithmetic wraps on
    int64 overflow, which is intentional for hash functions.
    """
    h = (xi.astype(_xp.int64) * _xp.int64(374761393)
         ^ yi.astype(_xp.int64) * _xp.int64(668265263)
         ^ _xp.int64(seed & 0x7FFF) * _xp.int64(2246822519))
    h = ((h ^ (h >> _xp.int64(15))) * _xp.int64(2246822519)) & _xp.int64(0x7FFFFFFF)
    h = ((h ^ (h >> _xp.int64(13))) * _xp.int64(3266489917)) & _xp.int64(0x7FFFFFFF)
    h ^= h >> _xp.int64(16)
    return h.astype(_xp.float64) / _xp.float64(0x7FFFFFFF) - 0.5


def _smooth(t):
    """Quintic smoothstep: 6t⁵ – 15t⁴ + 10t³  (C² continuous)."""
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _value_noise_at(wx, wy, scale: float, seed: int):
    """Sample smooth value noise at arbitrary float coordinates.

    Bilinearly interpolates a hash grid with quintic smoothing.
    Handles negative coordinates naturally via the floor/hash approach.
    """
    xs = wx / scale
    ys = wy / scale
    xi = _xp.floor(xs).astype(_xp.int64)
    yi = _xp.floor(ys).astype(_xp.int64)
    xf = xs - xi.astype(_xp.float64)
    yf = ys - yi.astype(_xp.float64)
    u, v = _smooth(xf), _smooth(yf)
    c00 = _hash(xi,     yi,     seed)
    c10 = _hash(xi + 1, yi,     seed)
    c01 = _hash(xi,     yi + 1, seed)
    c11 = _hash(xi + 1, yi + 1, seed)
    return (c00 * (1.0 - u) * (1.0 - v)
          + c10 * u          * (1.0 - v)
          + c01 * (1.0 - u) * v
          + c11 * u          * v)


def _fbm(wx, wy, scale: float, octaves: int,
         persistence: float, lacunarity: float, seed: int):
    """Fractional Brownian Motion at arbitrary float positions.

    Each successive octave doubles the frequency (lacunarity=2) and halves
    the amplitude (persistence=0.5), producing natural-looking multi-scale
    variation.  The result is normalised by the maximum possible amplitude
    so the output range stays near [-0.5, 0.5].
    """
    result = _xp.zeros(wx.shape, dtype=_xp.float64)
    amp = freq = 1.0
    max_amp = 0.0
    for i in range(octaves):
        result += amp * _value_noise_at(wx, wy, scale / freq, seed + i * 1337)
        max_amp += amp
        amp *= persistence
        freq *= lacunarity
    return result / max_amp


def _pixel_grid(width: int, height: int):
    """Return (xx, yy) meshgrid of shape (height, width)."""
    xs = _xp.arange(width,  dtype=_xp.float64)
    ys = _xp.arange(height, dtype=_xp.float64)
    return _xp.meshgrid(xs, ys)


def _normalise(arr) -> "_np_cpu.ndarray":
    """Linearly rescale array to [-1.0, 1.0] (float32 CPU numpy output)."""
    lo, hi = float(arr.min()), float(arr.max())
    if hi > lo:
        result = ((2.0 * (arr - lo) / (hi - lo)) - 1.0).astype(_xp.float32)
    else:
        result = _xp.zeros(arr.shape, dtype=_xp.float32)
    return _to_numpy(result)
