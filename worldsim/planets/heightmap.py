"""Procedural terrain and atmospheric generation for planet classes.

Three surface classes are supported:

**Continental**
    Full-quality heightmap generated via multi-octave fractional Brownian
    Motion with domain warp (ported from C# ``GenerateHeightMap``).  A
    biome overlay classifies each cell by terrain type; a resource heatmap
    is derived from both height and terrain.  Sea level is at 0.0.  Supports
    water systems, terrain deformation, and the full county/city placement
    pipeline.

**Single-biome terrestrial**
    Basic heightmap via ``GenerateBasicHeightMap`` (plain fBm, no domain
    warp) with a resource heatmap derived from height and the planet type
    only.  No biome variation; lower generation cost.  Covers desert worlds,
    ice worlds, and rocky/volcanic worlds.

**Gas giant / Ice giant**
    No surface.  An atmospheric resource heatmap stores hydrogen and
    helium-3 concentration bands together with storm-system metadata that
    affects extraction difficulty.  City placement is not supported; orbital
    extraction infrastructure will be added separately.
"""

from __future__ import annotations

from typing import Dict, List, Optional, TYPE_CHECKING

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

# Convenience aliases so the rest of the module reads as plain numpy.
np = _np_cpu  # used only for type-stable constants and random generators

if TYPE_CHECKING:
    from .infrastructure import Mine
    from .planet import Planet


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
# Core noise primitives
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Storm systems  (gas giant / ice giant)
# ---------------------------------------------------------------------------

def generate_storm_systems(width: int, height: int, seed: int) -> List[dict]:
    """Place 2–6 storm systems on a gas or ice giant.

    Each storm is a ``dict`` with keys:

    ``x``, ``y``
        Centre position in pixel-space.
    ``radius``
        Effective radius in pixels; cells within this distance are affected.
    ``intensity``
        Difficulty modifier in (0, 1]; higher values make extraction harder.

    Storm penalties can be evaluated via :func:`storm_extraction_penalty`.
    """
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 7))
    return [
        {
            "x":         float(rng.uniform(0, width)),
            "y":         float(rng.uniform(0, height)),
            "radius":    float(rng.uniform(5.0, 22.0)),
            "intensity": float(rng.uniform(0.3, 1.0)),
        }
        for _ in range(n)
    ]


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def storm_extraction_penalty(
    storm_systems: List[dict], x: float, y: float
) -> float:
    """Combined storm-driven extraction difficulty at pixel position (x, y).

    Returns a value in [0, 1]:  0 = no interference, 1 = maximum difficulty.
    Multiple overlapping storms are additive up to the cap.
    """
    penalty = 0.0
    for s in storm_systems:
        dist = ((x - s["x"]) ** 2 + (y - s["y"]) ** 2) ** 0.5
        if dist < s["radius"]:
            penalty += s["intensity"] * (1.0 - dist / s["radius"])
    return min(1.0, penalty)


# ---------------------------------------------------------------------------
# Mine terrain-deformation system
# ---------------------------------------------------------------------------

#: Ticks in Phase 1 (terrain flattening).  After this threshold the mine
#: transitions to Phase 2 (crater excavation).
MINE_PHASE1_DURATION: int   = 100

#: Phase 1 — broad, shallow Gaussian; flattens the landscape around the mine.
MINE_PHASE1_RADIUS:   float = 15.0   # pixels
MINE_PHASE1_AMOUNT:   float = -0.002  # height units applied per tick

#: Phase 2 — focused, deepening crater.  Depth per tick scales with output.
MINE_PHASE2_RADIUS: float = 8.0     # pixels
MINE_PHASE2_RATE:   float = 8e-5    # depth/tick per mine output unit

#: Heatmap depletion per tick per mine output unit (applied as Gaussian).
MINE_DEPLETION_RATE: float = 1e-4

#: Fraction of disrupted productive cells within crater radius that is
#: multiplied with this coefficient to produce a per-tick plague delta.
MINE_DISRUPTION_RATE: float = 0.10

#: Metal heatmap value below which the mine is considered exhausted.
MINE_EXHAUSTION_THRESHOLD: float = 0.02


def _radial_falloff(cx: float, cy: float, radius: float, H: int, W: int) -> "_np_cpu.ndarray":
    """Squared radial falloff kernel centred at ``(cx, cy)``.

    Returns a CPU float32 array of shape ``(H, W)`` with values in ``[0, 1]``.
    The falloff is ``max(0, 1 − dist/radius)^2``; it reaches 1.0 at the
    centre and drops to 0 at ``dist == radius``.
    """
    xs = _np_cpu.arange(W, dtype=_np_cpu.float32)
    ys = _np_cpu.arange(H, dtype=_np_cpu.float32)
    xx, yy = _np_cpu.meshgrid(xs, ys)
    dist = _np_cpu.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return _np_cpu.maximum(0.0, 1.0 - dist / max(float(radius), 1e-6)).astype(_np_cpu.float32) ** 2


def process_mine_turn(mine: "Mine", planet: "Planet") -> float:  # noqa: C901
    """Advance one simulation tick for *mine* and apply all terrain effects.

    The mine is aged, then the appropriate Gaussian deformation is applied to
    the planet's heightmap, the resource heatmap is depleted, and — for
    continental planets in Phase 2 — ecological disruption pressure is
    evaluated.

    Parameters
    ----------
    mine:
        The :class:`~.infrastructure.Mine` being processed.  Its
        :attr:`~.infrastructure.Mine.age` is incremented here; callers
        must **not** increment it separately.
    planet:
        The :class:`~.planet.Planet` the mine sits on.

    Returns
    -------
    float
        Plague level delta ≥ 0 to be added to ``planet.plague_level``.
        Only non-zero during Phase 2 when excavation is disrupting
        productive terrain on a continental planet.

    Notes
    -----
    Phase 1 (age 0–100)
        Large-radius (15 px), shallow (− 0.002/tick) Gaussian.  Terrain
        levels and flattens over 100 ticks, reducing construction costs for
        nearby structures (query with
        :meth:`~.planet.Planet.terrain_construction_bonus`).
    Phase 2 (age > 100)
        Smaller-radius (8 px), deepening Gaussian scaled by mine output.
        A visible crater develops.  Each tick the resource heatmap is
        locally depleted; when the metal concentration at the mine centre
        drops below :data:`MINE_EXHAUSTION_THRESHOLD`,
        ``mine.exhausted`` is set and the crater becomes inert.
    """
    mine.age += 1

    # No heightmap means no terrain effects (e.g. gas giants or legacy saves).
    if planet.heightmap is None:
        return 0.0

    H, W   = planet.heightmap.shape
    cx, cy = float(mine.x), float(mine.y)

    # ------------------------------------------------------------------
    # 1. Phase selection & terrain deformation
    # ------------------------------------------------------------------
    if mine.age <= MINE_PHASE1_DURATION:
        # Phase 1 — broad flattening
        radius = MINE_PHASE1_RADIUS
        amount = MINE_PHASE1_AMOUNT
    else:
        # Phase 2 — focused, deepening crater
        radius = MINE_PHASE2_RADIUS
        amount = -MINE_PHASE2_RATE * mine.output

    planet.deform_terrain(cx, cy, radius, amount)

    # Pre-compute the falloff once — reused for both depletion and ecology.
    falloff = _radial_falloff(cx, cy, radius, H, W)

    # ------------------------------------------------------------------
    # 2. Resource heatmap depletion
    # ------------------------------------------------------------------
    plague_delta = 0.0
    if planet.resource_heatmap:
        depletion = (mine.output * MINE_DEPLETION_RATE) * falloff
        for res in list(planet.resource_heatmap.keys()):
            planet.resource_heatmap[res] = np.maximum(
                0.0, planet.resource_heatmap[res] - depletion
            )
        # Check exhaustion at the mine’s centre pixel.
        py_idx = min(max(int(cy), 0), H - 1)
        px_idx = min(max(int(cx), 0), W - 1)
        metal = planet.resource_heatmap.get("metal")
        if metal is not None and float(metal[py_idx, px_idx]) < MINE_EXHAUSTION_THRESHOLD:
            mine.exhausted = True

    # ------------------------------------------------------------------
    # 3. Ecology disruption (Phase 2 + continental biome map only)
    # ------------------------------------------------------------------
    if mine.age > MINE_PHASE1_DURATION and planet.biome_map is not None:
        in_radius  = falloff > 0.0
        productive = np.isin(planet.biome_map, [TERRAIN_LOWLAND, TERRAIN_HIGHLAND])
        disrupted  = int(np.sum(in_radius & productive))
        area       = float(np.sum(in_radius))
        if disrupted > 0 and area > 0.0:
            plague_delta = MINE_DISRUPTION_RATE * (disrupted / area)

    return plague_delta


__all__ = [
    # Terrain constants
    "TERRAIN_DEEP_OCEAN",
    "TERRAIN_OCEAN",
    "TERRAIN_COAST",
    "TERRAIN_LOWLAND",
    "TERRAIN_HIGHLAND",
    "TERRAIN_MOUNTAIN",
    "TERRAIN_PEAK",
    "TERRAIN_NAMES",
    # Generators
    "generate_continental_heightmap",
    "generate_basic_heightmap",
    "generate_biome_map",
    "derive_resource_heatmap",
    "generate_atmospheric_heatmap",
    "generate_storm_systems",
    # Helpers
    "storm_extraction_penalty",
    # Mine terrain system
    "MINE_PHASE1_DURATION",
    "MINE_PHASE1_RADIUS",
    "MINE_PHASE1_AMOUNT",
    "MINE_PHASE2_RADIUS",
    "MINE_PHASE2_RATE",
    "MINE_DEPLETION_RATE",
    "MINE_DISRUPTION_RATE",
    "MINE_EXHAUSTION_THRESHOLD",
    "process_mine_turn",
]
