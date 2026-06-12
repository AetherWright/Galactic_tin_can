"""WorldSim package entry points.

The simulation is organised into arms — see the README architecture map:

``core``       backends, parallelism, geometry, runtime flags
``ai``         every learning component (Nelder-Mead, GRU+LoRA, GA, NEAT)
``galaxy``     stars, world genesis, initial settling
``planets``    surfaces, demographics, buildings, mining
``society``    culture, ideas, leaders, goals
``nations``    nation state, economy, government, construction, civilian AI
``military``   divisions, fleets, doctrine, combat, nuclear
``diplomacy``  messages, relations, war goals, peace, trade
``research``   the three-subsystem technology architecture
``events``     procedural events with Q-learned responses
``engine``     the century loop and world-level upkeep
``interface``  console presentation
"""

from .galaxy import init_world
from .engine import run_simulation, SimulationLoop
from .nations import Nation
from .society import Culture, Idea, get_idea, Leader, LeaderModel
from .planets import City, Spaceport, Planet
from .ai.persistence import save_gp_model, merge_and_save_models, load_model_seeds

__all__ = [
    "init_world",
    "run_simulation",
    "SimulationLoop",
    "City",
    "Planet",
    "Nation",
    "Culture",
    "Idea",
    "get_idea",
    "Leader",
    "LeaderModel",
    "Spaceport",
    "save_gp_model",
    "merge_and_save_models",
    "load_model_seeds",
]
