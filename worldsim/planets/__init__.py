"""Planetary models and registries.

The package exposes the pure Python :class:`Planet` implementation. Older
releases offered an optional Java helper, but that code has been removed in
favour of Rust and Python alternatives. ``set_use_java`` remains for backward
compatibility but no longer changes behaviour.
"""

from __future__ import annotations

from .registry import PLANETS
from .terrain import BIOMES
from .heightmap import (
    TERRAIN_DEEP_OCEAN,
    TERRAIN_OCEAN,
    TERRAIN_COAST,
    TERRAIN_LOWLAND,
    TERRAIN_HIGHLAND,
    TERRAIN_MOUNTAIN,
    TERRAIN_PEAK,
    TERRAIN_NAMES,
)

# The simulation now relies solely on the Python planet code.
USE_JAVA_PLANETS = False
from .planet import Planet


def set_use_java(flag: bool) -> None:  # noqa: D401 -- kept for API compatibility
    """No-op retained for compatibility."""
    global USE_JAVA_PLANETS
    USE_JAVA_PLANETS = False

from .city import City, process_city_batch
from .colony import Colony, process_colony_batch
from .county import County
from .infrastructure import (
    School,
    PowerPlant,
    ResearchLab,
    MilitaryBase,
    Mine,
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
    'USE_JAVA_PLANETS',
    'set_use_java',
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
    'Port',
    'Factory',
    'Hospital',
    'Shipyard',
    'Spaceport',
    'NuclearFacility',
    'OrbitalDefense',
]
