"""The nations arm: state, governance and internal development.

Submodules
----------
``nation``        the Nation dataclass and its per-fifth turn pipeline
``economy``       log-scale funds and resource stockpiles
``government``    government forms, approval and policy weights
``projects``      national project catalogue and progression
``construction``  resource collection and build actions on planet surfaces
``civilian``      the departmental civilian AI action space and controller
"""

from .civilian import (
    ACTION_GROUPS,
    CivilianController,
    CivilianDepartment,
    DEPARTMENTS,
    N_CIVILIAN_ACTIONS,
    N_CIVILIAN_DEPARTMENTS,
    process_action_queue,
)
from .construction import RESOURCE_COSTS
from .economy import Economy
from .government import Government
from .nation import Nation
from .projects import (
    NationalProject,
    PROJECT_CATALOG,
    PROJECT_NAMES,
    ProjectSpec,
)

__all__ = [
    "Nation",
    "Economy",
    "Government",
    "NationalProject",
    "ProjectSpec",
    "PROJECT_CATALOG",
    "PROJECT_NAMES",
    "RESOURCE_COSTS",
    "CivilianController",
    "CivilianDepartment",
    "DEPARTMENTS",
    "ACTION_GROUPS",
    "N_CIVILIAN_ACTIONS",
    "N_CIVILIAN_DEPARTMENTS",
    "process_action_queue",
]
