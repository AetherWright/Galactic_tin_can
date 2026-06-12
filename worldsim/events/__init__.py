"""The events arm: random and procedural happenings with learned responses.

Submodules
----------
``catalog``    static flavour events
``generator``  situation-aware procedural event generation
``qlearner``   contextual Q-table learning event responses
``engine``     trigger probability, choice resolution and persistence
"""

from .catalog import EVENTS
from .engine import EventDecisionEngine
from .generator import ProceduralEventGenerator
from .qlearner import EventQLearner

__all__ = [
    "EVENTS",
    "EventDecisionEngine",
    "ProceduralEventGenerator",
    "EventQLearner",
]
