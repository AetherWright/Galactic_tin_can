"""Biome definitions and resource generation utilities.

This module enumerates the basic planetary biomes and provides helper
functions to seed starting resource quantities for a newly generated
planet.  Biomes influence the distribution of ``food``, ``metal`` and
``energy`` resources which nations can later exploit.
"""

from __future__ import annotations

from typing import Dict
import random

from ..config import load_json

# Resource ranges for each biome type.  Values are stored in an external
# configuration file to keep runtime memory usage low.
_RAW_BIOMES = load_json("biome_resources")
BIOME_RESOURCES: Dict[str, Dict[str, tuple[int, int]]] = {
    biome: {res: (rng[0], rng[1]) for res, rng in resources.items()}
    for biome, resources in _RAW_BIOMES.items()
}

# Public list of available biome names.
BIOMES = list(BIOME_RESOURCES.keys())

# Maximum food storage capacity per biome (cap = upper bound of food range).
BIOME_FOOD_CAP: Dict[str, float] = {
    biome: float(res["food"][1])
    for biome, res in BIOME_RESOURCES.items()
    if "food" in res
}

# Multiplier applied to Farm.output per biome.
# Forest/oceanic = fertile farming; desert = irrigation-dependent;
# ice/volcanic = near-subsistence only.
BIOME_FARM_YIELD: Dict[str, float] = {
    "forest":   1.4,
    "oceanic":  1.2,
    "desert":   0.5,
    "ice":      0.3,
    "volcanic": 0.2,
}


def generate_resources(biome: str) -> Dict[str, float]:
    """Return starting resource amounts for ``biome``.

    Parameters
    ----------
    biome:
        Name of the biome whose resources should be generated.

    Returns
    -------
    Dict[str, float]
        Mapping of resource name to initial quantity.
    """

    spec = BIOME_RESOURCES[biome]
    return {res: float(random.randint(rng[0], rng[1])) for res, rng in spec.items()}


__all__ = ["BIOMES", "BIOME_RESOURCES", "BIOME_FARM_YIELD", "BIOME_FOOD_CAP", "generate_resources"]

