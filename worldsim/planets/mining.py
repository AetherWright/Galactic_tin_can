"""Mine terrain-deformation system.

Mines age through two phases: a broad flattening phase and a crater
excavation phase that depletes the local resource heatmap and can disrupt
the surrounding ecology.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy as _np_cpu

from .surface.heightmaps import TERRAIN_HIGHLAND, TERRAIN_LOWLAND

if TYPE_CHECKING:
    from .buildings import Mine
    from .planet import Planet


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
        The :class:`~.buildings.Mine` being processed.  Its
        :attr:`~.buildings.Mine.age` is incremented here; callers
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
