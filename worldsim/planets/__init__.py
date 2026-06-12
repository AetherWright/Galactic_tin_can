"""The planetary arm: surfaces, settlements and population dynamics.

Submodules
----------
``registry``      the global ``PLANETS`` registry
``planet``        planet state and the surface/settlement pipeline
``terrain``       biome definitions and biome resource tables
``surface``       procedural heightmaps, resource heatmaps and storms
``mining``        mine aging, terrain deformation and depletion
``demographics``  city / colony / county population models
``buildings``     physical structures placed on planet surfaces
"""

from __future__ import annotations

from .registry import PLANETS
from .terrain import BIOMES
from .surface import (
    TERRAIN_DEEP_OCEAN,
    TERRAIN_OCEAN,
    TERRAIN_COAST,
    TERRAIN_LOWLAND,
    TERRAIN_HIGHLAND,
    TERRAIN_MOUNTAIN,
    TERRAIN_PEAK,
    TERRAIN_NAMES,
)
from .planet import Planet
from .demographics import (
    City,
    Colony,
    County,
    process_city_batch,
    process_colony_batch,
)
from .buildings import (
    School,
    PowerPlant,
    ResearchLab,
    MilitaryBase,
    Mine,
    Farm,
    Port,
    Factory,
    Hospital,
    Shipyard,
    Spaceport,
    NuclearFacility,
    OrbitalDefense,
)

__all__ = [
    'PLANETS',
    'Planet',
    'BIOMES',
    'TERRAIN_DEEP_OCEAN',
    'TERRAIN_OCEAN',
    'TERRAIN_COAST',
    'TERRAIN_LOWLAND',
    'TERRAIN_HIGHLAND',
    'TERRAIN_MOUNTAIN',
    'TERRAIN_PEAK',
    'TERRAIN_NAMES',
    'City',
    'process_city_batch',
    'Colony',
    'process_colony_batch',
    'County',
    'School',
    'PowerPlant',
    'ResearchLab',
    'MilitaryBase',
    'Mine',
    'Farm',
    'Port',
    'Factory',
    'Hospital',
    'Shipyard',
    'Spaceport',
    'NuclearFacility',
    'OrbitalDefense',
]
