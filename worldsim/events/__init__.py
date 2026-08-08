"""The events arm: random and procedural happenings with learned responses.

Submodules
----------
``catalog``    static flavour events
``generator``  situation-aware procedural event generation
``qlearner``   legacy per-event tabular Q-learning (no-torch fallback only —
               see ``worldsim.ai.rnn.roles.TorchEventAI`` for the shared
               categorical agent used when torch is available)
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
