"""Storm systems on gas and ice giants and their extraction penalties."""

from __future__ import annotations

from typing import List

from .noise import np


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
