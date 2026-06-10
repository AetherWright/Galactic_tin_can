"""The engine arm: the century loop and world-level upkeep.

Submodules
----------
``loop``       ``run_simulation`` and the step-wise ``SimulationLoop``
``scoring``    per-century MetaGA fitness scoring
``reporting``  console summaries
``territory``  planet upkeep and star ownership
``blocs``      alliance bloc computation and border pressure
``politics``   collapses, succession states and civil wars
``filters``    great filters and zombie-nation culling
"""

from .blocs import compute_alliance_blocs, update_border_pressures
from .loop import SimulationLoop, run_simulation
from .scoring import apply_century_scores, century_score
from .territory import process_planets, update_star_ownership

__all__ = [
    "run_simulation",
    "SimulationLoop",
    "compute_alliance_blocs",
    "update_border_pressures",
    "process_planets",
    "update_star_ownership",
    "apply_century_scores",
    "century_score",
]
