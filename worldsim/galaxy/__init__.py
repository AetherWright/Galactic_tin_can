"""The galaxy arm: astrography and world genesis.

Submodules
----------
``star``      Star system model and the global ``STARS`` registry
``genesis``   procedural star-cluster generation and ``init_world``
``settling``  initial nation placement on starting colonies
"""

from .genesis import (
    add_star_cluster,
    add_star_systems,
    get_next_nation_name,
    init_world,
)
from .settling import assign_initial_cities
from .star import Star, STARS

__all__ = [
    "Star",
    "STARS",
    "init_world",
    "add_star_cluster",
    "add_star_systems",
    "get_next_nation_name",
    "assign_initial_cities",
]
