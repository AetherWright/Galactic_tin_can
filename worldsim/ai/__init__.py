"""The AI arm: every learning and decision-making component.

Submodules
----------
``perceptron``       SimplePerceptron (vectorised on NumPy/CuPy)
``networks``         LayeredNetwork — sparse bidirectional layered net (NumPy/CuPy)
``nelder_mead``      plain simplex minimiser
``policy``           NelderMeadPolicy base class
``roles``            per-role Nelder-Mead facades (WarAI, DiplomacyAI, …)
``strategic``        StrategicNelderMeadAI continuous planner
``rnn``              PyTorch GRU base models + per-nation LoRA adapters
                     (war, civilian, fleet, … role controllers)
``native``           Rust/C++ perceptron banks for division controllers
``neat``/``graph``   NEAT-inspired topology evolution
``meta_ga``          reward-weight genetic algorithm
``embeddings``       action embedding helpers
``representations``  galactic state tensors fed to controllers
``persistence``      cross-run merging and seeding of trained models
"""

from __future__ import annotations

from .nelder_mead import SimplexResult, nelder_mead_minimise
from .networks import LayeredNetwork
from .perceptron import SimplePerceptron
from .policy import NelderMeadPolicy
from .roles import (
    CivilianOverseerAI,
    DepartmentPolicyAI,
    DiplomacyAI,
    DomesticPolicyAI,
    ProjectAI,
    ResearchAI,
    WarAI,
)
from .strategic import NelderMeadResult, StrategicNelderMeadAI

# ---------------------------------------------------------------------------
# Compatibility flags
# ---------------------------------------------------------------------------

_USE_NEAT: bool = False
_USE_RUST: bool = False


def set_use_neat(flag: bool) -> None:
    """Retain the old API for toggling NEAT usage."""

    global _USE_NEAT
    _USE_NEAT = bool(flag)


def set_use_rust(flag: bool) -> None:
    """Toggle the native (Rust/C++) perceptron backend for division controllers."""

    global _USE_RUST
    _USE_RUST = bool(flag)
    from . import native as _native
    _native.set_enabled(_USE_RUST)


__all__ = [
    "SimplexResult",
    "nelder_mead_minimise",
    "LayeredNetwork",
    "SimplePerceptron",
    "NelderMeadPolicy",
    "WarAI",
    "DomesticPolicyAI",
    "CivilianOverseerAI",
    "DepartmentPolicyAI",
    "ProjectAI",
    "DiplomacyAI",
    "ResearchAI",
    "NelderMeadResult",
    "StrategicNelderMeadAI",
    "set_use_neat",
    "set_use_rust",
]
