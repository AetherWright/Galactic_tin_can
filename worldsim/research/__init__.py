"""The research arm: the three-subsystem technology architecture.

A layered research model:

* Three :class:`~worldsim.research.subsystems.TechnologySubsystem` instances —
  physics, engineering and biology.  Each owns its own named technology
  graph, its own :class:`~worldsim.ai.ResearchAI` and its own research-point
  pool.
* A :class:`~worldsim.research.director.ResearchDirector` that receives the
  nation's total research points each tick, allocates them between the
  subsystems and is aware of cross-subsystem prerequisites.

Technologies publish their effects through ``nation.tech_bonuses``; every
effect is a first-class callable stored on the
:class:`~worldsim.research.technology.TechnologyNode`.

Submodules
----------
``technology``  graph primitives (Technology, TechnologyNode, TechnologyTree)
``effects``     unlock-effect callables
``subsystems``  the physics / engineering / biology research domains
``director``    point allocation across subsystems
"""

from .director import ResearchDirector, setup_default_tech_tree
from .subsystems import (
    BiologySubsystem,
    EngineeringSubsystem,
    PhysicsSubsystem,
    TechnologySubsystem,
)
from .technology import (
    Technology,
    TechnologyNode,
    TechnologyTree,
    add_random_technologies,
    generate_random_technology,
    generate_tech_name,
)

__all__ = [
    "Technology",
    "TechnologyNode",
    "TechnologyTree",
    "TechnologySubsystem",
    "PhysicsSubsystem",
    "EngineeringSubsystem",
    "BiologySubsystem",
    "ResearchDirector",
    "generate_tech_name",
    "generate_random_technology",
    "add_random_technologies",
    "setup_default_tech_tree",
]
