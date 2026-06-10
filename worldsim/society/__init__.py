"""The society arm: cultural identity shared by nations and their people.

Deliberately lightweight — these modules are imported by both the planetary
demographics layer and the nation layer, so they must not depend on either.

Submodules
----------
``culture``  trait vectors and archetype bonuses
``ideas``    national ideas adopted by cities and counties
``leader``   leader personalities and the leader reward model
``goals``    dynamic goal management
"""

from .culture import ARCHETYPE_BONUSES, ARCHETYPE_IDEALS, ArchetypeBonus, Culture
from .goals import Goal, GoalManager, GoalTemplate
from .ideas import Idea, get_idea
from .leader import Leader, LeaderModel

__all__ = [
    "Culture",
    "ArchetypeBonus",
    "ARCHETYPE_BONUSES",
    "ARCHETYPE_IDEALS",
    "Idea",
    "get_idea",
    "Leader",
    "LeaderModel",
    "Goal",
    "GoalManager",
    "GoalTemplate",
]
