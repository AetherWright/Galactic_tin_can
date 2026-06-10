"""Core foundations shared by every arm of the simulation.

Submodules
----------
``backend``    NumPy/CuPy array backend selection
``native``     optional Rust/C++ accelerator loading
``flags``      runtime verbosity/approximation toggles
``geometry``   distance, travel time and polygon helpers
``growth``     logistic growth
``parallel``   shared process/thread pools
``timing``     wall-clock guards

The package re-exports the frequently used names so call sites can simply do
``from worldsim.core import distance, vprint``.  Note that the mutable flags
(``VERBOSE`` …) must be read via ``from worldsim.core import flags`` to see
runtime updates.
"""

from .backend import _np, USING_GPU, as_array, get_array_module
from .flags import (
    APPROXIMATE,
    DEBUG_PROFILE,
    VERBOSE,
    WATCHED_NATION,
    _compact_pop,
    vprint,
    wprint,
)
from .geometry import (
    LAND_SPEED_KM_PER_DAY,
    SPACE_SPEED_KM_PER_DAY,
    distance,
    distance_sq,
    polygon_area,
    polygon_centroid,
    travel_time,
)
from .growth import logistic_growth
from .native import USE_RUST, set_use_rust
from .parallel import gather_batches, mp_map, pooled_map, shutdown_pool, tp_map
from .timing import time_limit

__all__ = [
    "_np",
    "USING_GPU",
    "as_array",
    "get_array_module",
    "APPROXIMATE",
    "DEBUG_PROFILE",
    "VERBOSE",
    "WATCHED_NATION",
    "_compact_pop",
    "vprint",
    "wprint",
    "LAND_SPEED_KM_PER_DAY",
    "SPACE_SPEED_KM_PER_DAY",
    "distance",
    "distance_sq",
    "polygon_area",
    "polygon_centroid",
    "travel_time",
    "logistic_growth",
    "USE_RUST",
    "set_use_rust",
    "gather_batches",
    "mp_map",
    "pooled_map",
    "shutdown_pool",
    "tp_map",
    "time_limit",
]
