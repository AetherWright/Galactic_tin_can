"""Resource and atmospheric heatmap derivation."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as _np_cpu  # noqa: F401

from .noise import _fbm, _pixel_grid, _to_numpy, _value_noise_at, _xp, np


# ---------------------------------------------------------------------------
# Resource heatmaps
# ---------------------------------------------------------------------------

def _heatmap_continental(h: "_np_cpu.ndarray", _bmap,
                          seed: int) -> Dict[str, "_np_cpu.ndarray"]:
    """Per-resource intensity for a continental planet."""
    H, W = h.shape
    xx, yy = _pixel_grid(W, H)
    h = _xp.asarray(h).astype(_xp.float64)

    metal   = _xp.clip(_xp.maximum(0.0, h) * 0.5
                     + _xp.maximum(0.0, h - 0.6) * 1.5, 0.0, 1.0)
    food    = _xp.where(h < 0.0, 0.0,
                        _xp.clip(_xp.minimum(h + 0.3, 0.8 - h) * 3.0, 0.0, 1.0))
    coast   = _xp.clip((0.15 - _xp.abs(h)) * 5.0, 0.0, 1.0)
    e_noise = _value_noise_at(xx, yy, 25.0, seed + 30_000) * 0.15 + 0.10
    energy  = _xp.clip(coast * 0.6 + e_noise, 0.0, 1.0)
    u_noise = (_fbm(xx, yy, 20.0, 3, 0.5, 2.0, seed + 40_000) + 1.0) * 0.5
    uranium = _xp.where(h > 0.7,
                        u_noise * _xp.clip((h - 0.7) * 5.0, 0.0, 1.0),
                        0.0)

    return {
        "metal":   _to_numpy(metal.astype(_xp.float32)),
        "food":    _to_numpy(food.astype(_xp.float32)),
        "energy":  _to_numpy(energy.astype(_xp.float32)),
        "uranium": _to_numpy(uranium.astype(_xp.float32)),
    }


def _heatmap_terrestrial(h: "_np_cpu.ndarray", planet_type: str,
                          seed: int) -> Dict[str, "_np_cpu.ndarray"]:
    """Per-resource intensity for a single-biome terrestrial planet.

    Each planet type has a characteristic resource profile:

    * **Rocky / Volcanic** — very high metal, geothermal energy, high uranium.
    * **Desert world**     — abundant solar energy, moderate metal, negligible food.
    * **Ice world**        — subsurface uranium hotspots, moderate metal, no food.
    * **Generic**          — balanced baseline used for plain ``"terrestrial"`` worlds.
    """
    H, W = h.shape
    xx, yy = _pixel_grid(W, H)
    h = _xp.asarray(h).astype(_xp.float64)
    n = (_fbm(xx, yy, 30.0, 3, 0.5, 2.0, seed) + 1.0) * 0.5  # noise in [0, 1]

    if planet_type in {"rocky", "volcanic"}:
        metal   = _xp.clip(0.45 + h * 0.40 + n * 0.15, 0.0, 1.0)
        energy  = _xp.clip(n * 0.75 + _xp.maximum(0.0, -h) * 0.3, 0.0, 1.0)
        food    = _xp.zeros(h.shape, dtype=_xp.float64)
        uranium = _xp.clip(n * 0.55 + _xp.maximum(0.0, h - 0.5) * 0.45, 0.0, 1.0)

    elif planet_type == "desert_world":
        energy  = _xp.clip(0.60 + n * 0.30, 0.0, 1.0)
        metal   = _xp.clip(0.20 + n * 0.40 + h * 0.10, 0.0, 1.0)
        food    = _xp.clip(_xp.maximum(0.0, 0.08 - h * 0.15) * 0.6, 0.0, 0.10)
        uranium = _xp.clip(n * 0.30 + _xp.maximum(0.0, h - 0.4) * 0.30, 0.0, 0.60)

    elif planet_type == "ice_world":
        # Subsurface uranium hotspots driven by a second independent noise pass
        sub     = (_fbm(xx, yy, 15.0, 2, 0.6, 2.0, seed + 500) + 1.0) * 0.25
        uranium = _xp.clip(n * 0.45 + sub, 0.0, 1.0)
        metal   = _xp.clip(0.25 + n * 0.30, 0.0, 0.70)
        food    = _xp.zeros(h.shape, dtype=_xp.float64)
        energy  = _xp.clip(0.08 + n * 0.18, 0.0, 0.30)

    else:  # generic terrestrial  (HZ world with random biome)
        metal   = _xp.clip(0.25 + n * 0.45, 0.0, 1.0)
        energy  = _xp.clip(0.30 + n * 0.30, 0.0, 1.0)
        food    = _xp.clip(_xp.maximum(0.0, 0.5 - h * 0.5) * 0.8, 0.0, 1.0)
        uranium = _xp.clip(n * 0.20, 0.0, 0.40)

    return {
        "metal":   _to_numpy(metal.astype(_xp.float32)),
        "food":    _to_numpy(food.astype(_xp.float32)),
        "energy":  _to_numpy(energy.astype(_xp.float32)),
        "uranium": _to_numpy(uranium.astype(_xp.float32)),
    }


def derive_resource_heatmap(
    heightmap: np.ndarray,
    biome_map: Optional[np.ndarray],
    planet_type: str,
    seed: int,
) -> Dict[str, np.ndarray]:
    """Return per-resource spatial intensity maps for *planet_type*.

    ``biome_map`` is consulted only for continental planets.  For all other
    surface types the ``planet_type`` string drives the derivation.
    """
    if planet_type == "continental":
        return _heatmap_continental(heightmap, biome_map, seed)
    return _heatmap_terrestrial(heightmap, planet_type, seed)


# ---------------------------------------------------------------------------
# Atmospheric heatmap  (gas giant / ice giant)
# ---------------------------------------------------------------------------

def generate_atmospheric_heatmap(
    width: int, height: int, seed: int
) -> Dict[str, "_np_cpu.ndarray"]:
    """Banded atmospheric concentration maps for gas and ice giants."""
    # Use CPU numpy for the random generator and 1-D arrays;
    # use _xp for the 2-D noise pass so it benefits from GPU when available.
    rng = _np_cpu.random.default_rng(seed)
    xx, yy = _pixel_grid(width, height)

    lat = _np_cpu.arange(height, dtype=_np_cpu.float64) / height  # normalised [0, 1]

    # Hydrogen — continuous banding with mild noise turbulence
    freq   = float(rng.uniform(3.0, 7.0))
    h2_1d  = 0.6 + 0.4 * _np_cpu.cos(2.0 * _np_cpu.pi * freq * lat)
    h2_n   = _value_noise_at(xx, yy, 30.0, int(rng.integers(0, 100_000))) * 0.12
    h2_1d_xp = _xp.asarray(h2_1d)
    hydrogen = _xp.clip(h2_1d_xp[:, _np_cpu.newaxis] + h2_n, 0.0, 1.0).astype(_xp.float32)

    # Helium-3 — narrow Gaussian bands at randomly chosen latitudes
    n_bands  = int(rng.integers(2, 5))
    he3_1d   = _np_cpu.zeros(height, dtype=_np_cpu.float64)
    for _ in range(n_bands):
        centre = float(rng.uniform(0.1, 0.9))
        sigma  = float(rng.uniform(4.0, 12.0)) / height
        peak   = float(rng.uniform(0.3, 0.8))
        he3_1d += peak * _np_cpu.exp(-0.5 * ((lat - centre) / sigma) ** 2)
    he3_n  = _value_noise_at(xx, yy, 20.0, int(rng.integers(0, 100_000))) * 0.08
    he3_1d_xp = _xp.asarray(he3_1d)
    helium3 = _xp.clip(he3_1d_xp[:, _np_cpu.newaxis] + he3_n, 0.0, 1.0).astype(_xp.float32)

    return {"hydrogen": _to_numpy(hydrogen), "helium3": _to_numpy(helium3)}
