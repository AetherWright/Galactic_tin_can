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

Submodules
----------
``noise``       hash/value-noise/fBm primitives (GPU-aware)
``heightmaps``  heightmap generators and the ``TERRAIN_*`` classification
``resources``   resource and atmospheric heatmap derivation
``weather``     storm systems and extraction penalties
"""

from .heightmaps import (
    TERRAIN_COAST,
    TERRAIN_DEEP_OCEAN,
    TERRAIN_HIGHLAND,
    TERRAIN_LOWLAND,
    TERRAIN_MOUNTAIN,
    TERRAIN_NAMES,
    TERRAIN_OCEAN,
    TERRAIN_PEAK,
    generate_basic_heightmap,
    generate_biome_map,
    generate_continental_heightmap,
)
from .resources import derive_resource_heatmap, generate_atmospheric_heatmap
from .weather import generate_storm_systems, storm_extraction_penalty

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
]
