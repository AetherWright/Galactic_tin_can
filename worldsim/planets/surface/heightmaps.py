"""Heightmap generation and terrain classification.

See the package docstring of :mod:`worldsim.planets.surface` for the three
supported surface classes.
"""

from __future__ import annotations

from typing import Dict

import numpy as _np_cpu  # noqa: F401

from .noise import _fbm, _normalise, _pixel_grid, _to_numpy, _xp


# ---------------------------------------------------------------------------
# Terrain type constants (biome_map integer codes)
# ---------------------------------------------------------------------------

TERRAIN_DEEP_OCEAN: int = 0
TERRAIN_OCEAN:      int = 1
TERRAIN_COAST:      int = 2
TERRAIN_LOWLAND:    int = 3
TERRAIN_HIGHLAND:   int = 4
TERRAIN_MOUNTAIN:   int = 5
TERRAIN_PEAK:       int = 6

TERRAIN_NAMES: Dict[int, str] = {
    TERRAIN_DEEP_OCEAN: "deep_ocean",
    TERRAIN_OCEAN:      "ocean",
    TERRAIN_COAST:      "coast",
    TERRAIN_LOWLAND:    "lowland",
    TERRAIN_HIGHLAND:   "highland",
    TERRAIN_MOUNTAIN:   "mountain",
    TERRAIN_PEAK:       "peak",
}


# ---------------------------------------------------------------------------
# Continental heightmap  (GenerateHeightMap port)
# ---------------------------------------------------------------------------

def generate_continental_heightmap(width: int, height: int, seed: int) -> "_np_cpu.ndarray":
    """Multi-octave fBm with domain warp — full GenerateHeightMap quality.

    Domain warp is applied by sampling a pair of lower-frequency fBm fields
    as warp offsets before evaluating the main 6-octave layer.  This breaks
    the regular grid patterns that plain fBm produces and creates more
    natural-looking coastlines, mountain ridges, and river valleys.

    Parameters
    ----------
    width, height:
        Pixel dimensions of the output array.
    seed:
        RNG seed; the same seed always produces the same terrain.

    Returns
    -------
    np.ndarray
        float32 of shape ``(height, width)``, values in ``[-1.0, 1.0]``.
        Sea level is at ``0.0``.
    """
    xx, yy = _pixel_grid(width, height)
    # Warp offset fields — lower frequency, fewer octaves than the main layer
    warp_x = _fbm(xx, yy, 80.0, 3, 0.5, 2.0, seed + 10_000) * 25.0
    warp_y = _fbm(xx, yy, 80.0, 3, 0.5, 2.0, seed + 20_000) * 25.0
    # Main layer sampled at warped pixel positions
    h = _fbm(xx + warp_x, yy + warp_y, 50.0, 6, 0.5, 2.0, seed)
    return _normalise(h)


# ---------------------------------------------------------------------------
# Basic heightmap  (GenerateBasicHeightMap)
# ---------------------------------------------------------------------------

def generate_basic_heightmap(width: int, height: int, seed: int) -> "_np_cpu.ndarray":
    """Plain multi-octave fBm without domain warp — GenerateBasicHeightMap.

    Lower generation cost than the continental variant; used for single-biome
    terrestrial worlds (desert, ice, rocky/volcanic).

    Returns
    -------
    np.ndarray
        float32 of shape ``(height, width)``, values in ``[-1.0, 1.0]``.
    """
    xx, yy = _pixel_grid(width, height)
    h = _fbm(xx, yy, 40.0, 4, 0.5, 2.0, seed)
    return _normalise(h)


# ---------------------------------------------------------------------------
# Biome overlay  (continental only)
# ---------------------------------------------------------------------------

def generate_biome_map(heightmap: "_np_cpu.ndarray") -> "_np_cpu.ndarray":
    """Classify each cell into a ``TERRAIN_*`` type from a continental heightmap.

    Classification thresholds (sea level = 0.0):

    ============  ==============================
    Type          Height range
    ============  ==============================
    DEEP_OCEAN    h < −0.5
    OCEAN         −0.5 ≤ h < 0
    COAST         |h| < 0.05  (overrides above)
    LOWLAND       0 ≤ h < 0.4
    HIGHLAND      0.4 ≤ h < 0.7
    MOUNTAIN      0.7 ≤ h < 0.85
    PEAK          h ≥ 0.85
    ============  ==============================

    The coastal strip is assigned last so it overrides both the shallow-ocean
    and near-shore lowland edges, giving a consistent thin coastal band.

    Returns
    -------
    np.ndarray
        uint8 of shape ``(height, width)`` with ``TERRAIN_*`` values.
    """
    # Operate on the active array module for potential GPU acceleration,
    # then return a CPU numpy array so callers don't need to handle CuPy.
    h = _xp.asarray(heightmap)
    bmap = _xp.full(h.shape, TERRAIN_LOWLAND, dtype=_xp.uint8)
    bmap[h < -0.5]                   = TERRAIN_DEEP_OCEAN
    bmap[(h >= -0.5) & (h < 0.0)]   = TERRAIN_OCEAN
    bmap[(h >= 0.4)  & (h < 0.7)]   = TERRAIN_HIGHLAND
    bmap[(h >= 0.7)  & (h < 0.85)]  = TERRAIN_MOUNTAIN
    bmap[h >= 0.85]                  = TERRAIN_PEAK
    # Thin coastal band written last so it straddles the waterline cleanly
    bmap[_xp.abs(h) < 0.05]         = TERRAIN_COAST
    return _to_numpy(bmap)
