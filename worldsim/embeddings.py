"""Action embedding utilities.

Provide simple numeric vectors describing actions. These can be
extended for learning algorithms that require more context than a
categorical label.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

from .utils import get_array_module


ACTION_FEATURE_LABELS: List[str] = [
    "economic_cost",
    "resource_cost",
    "population_gain",
    "military_gain",
    "science_gain",
    "infrastructure_gain",
    "diplomacy_gain",
    "environmental_impact",
]


@dataclass
class ActionEmbedding:
    """Mapping between an action name, its index and feature vector."""

    name: str
    index: int
    vector: List[float]

    @property
    def one_hot(self) -> List[float]:
        """Return a one-hot vector representing the discrete action index."""

        hot = [0.0] * len(_ACTION_NAMES)
        if 0 <= self.index < len(hot):
            hot[self.index] = 1.0
        return hot

    @property
    def features(self) -> Dict[str, float]:
        """Return a mapping between feature labels and their values."""

        return dict(zip(ACTION_FEATURE_LABELS, self.vector))

    def as_vector(self, include_discrete: bool = False) -> List[float]:
        """Return feature vector optionally concatenated with the one-hot code."""

        base = list(self.vector)
        if include_discrete:
            base.extend(self.one_hot)
        return base


# Basic feature vectors follow ``ACTION_FEATURE_LABELS`` ordering.
_ACTION_VECTORS: Dict[str, List[float]] = {
    "build_city": [0.5, 0.1, 1.0, 0.0, 0.0, 0.8, 0.2, -0.1],
    "build_mine": [0.3, 0.3, 0.0, 0.0, 0.0, 0.3, 0.0, -0.3],
    "build_factory": [0.5, 0.5, 0.0, 0.0, 0.0, 0.7, 0.0, -0.2],
    "build_lab": [0.4, 0.3, 0.0, 0.0, 1.0, 0.5, 0.1, -0.1],
    "research_tech": [0.2, 0.0, 0.0, 0.0, 1.0, 0.2, 0.1, 0.0],
    "produce_nukes": [0.6, 0.7, 0.0, 1.0, 0.0, 0.2, -0.4, -0.6],
    "build_nuke_facility": [0.5, 0.5, 0.0, 1.0, 0.0, 0.3, -0.3, -0.5],
    "build_base": [0.6, 0.2, 0.0, 1.0, 0.0, 0.4, -0.1, -0.2],
    "build_orbital_defense": [0.6, 0.4, 0.0, 1.0, 0.0, 0.6, -0.1, -0.3],
    "build_port": [0.4, 0.2, 0.0, 0.0, 0.0, 0.6, 0.2, -0.1],
    "build_hospital": [0.4, 0.2, 0.0, 0.0, 0.0, 0.5, 0.1, 0.3],
    "build_shipyard": [0.5, 0.3, 0.0, 1.0, 0.0, 0.5, -0.1, -0.2],
    "build_school": [0.3, 0.2, 0.0, 0.0, 0.2, 0.4, 0.3, 0.2],
    "build_power_plant": [0.4, 0.3, 0.0, 0.0, 0.0, 0.7, -0.1, -0.4],
    "build_spaceport": [0.6, 0.4, 0.0, 0.0, 0.1, 0.9, 0.1, -0.3],
    "build_division": [0.4, 0.3, -1.0, 1.0, 0.0, 0.1, -0.3, -0.5],
    "colonize_planet": [0.5, 0.5, 1.0, 0.0, 0.0, 0.9, 0.2, -0.2],
    "upgrade_assets": [0.3, 0.2, 0.3, 0.2, 0.2, 0.6, 0.1, 0.0],
    "start_project": [0.7, 0.4, 0.3, 0.3, 0.3, 0.8, 0.2, -0.1],
    "launch_first_strike": [0.9, 0.5, -1.0, 1.0, 0.0, 0.1, -0.6, -0.9],
    "launch_nuclear_strike": [0.9, 0.5, -0.8, 0.9, 0.0, 0.1, -0.6, -1.0],
    "military_order_attack": [0.2, 0.0, -0.1, 0.7, 0.0, 0.1, -0.2, -0.4],
    "military_order_defend": [0.1, 0.0, 0.0, 0.5, 0.0, 0.1, 0.0, -0.2],
    "military_order_reserve": [0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.1, -0.1],
    "diplomacy_idle": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "diplomacy_form_alliance": [0.2, 0.0, 0.3, 0.2, 0.0, 0.2, 0.8, 0.2],
    "diplomacy_declare_war": [0.4, 0.0, -0.2, 0.8, 0.0, 0.1, -0.4, -0.6],
    "Highway Network": [0.7, 0.3, 0.4, 0.2, 0.1, 1.0, 0.3, -0.2],
    "Research Complex": [0.6, 0.2, 0.1, 0.1, 0.8, 0.6, 0.2, 0.0],
    "Orbital Defense Grid": [0.8, 0.3, 0.0, 0.9, 0.1, 0.7, -0.2, -0.4],
    "Mega Dam": [0.7, 0.4, 0.5, 0.2, 0.1, 0.9, 0.1, -0.3],
    "AI Governance System": [0.8, 0.3, 0.1, 0.3, 0.4, 0.7, 0.5, -0.1],
    "Resilience Program": [0.9, 0.3, 0.2, 0.1, 0.2, 0.6, 0.6, 0.3],
    "Orbital Shipyard": [0.9, 0.4, 0.1, 1.0, 0.0, 0.8, -0.2, -0.3],
}

_DEFAULT_VECTOR: List[float] = [0.0] * len(ACTION_FEATURE_LABELS)
_ACTION_NAMES: List[str] = list(_ACTION_VECTORS.keys())
_ACTION_INDEX: Dict[str, int] = {name: idx for idx, name in enumerate(_ACTION_NAMES)}


def list_action_names() -> List[str]:
    """Return the ordered list of known action names."""

    return list(_ACTION_NAMES)


def get_action_index(name: str) -> int:
    """Return the discrete index for ``name`` or ``-1`` if unknown."""

    return _ACTION_INDEX.get(name, -1)


def get_action_one_hot(name: str) -> List[float]:
    """Return a one-hot encoding for ``name`` within the known action space."""

    idx = get_action_index(name)
    hot = [0.0] * len(_ACTION_NAMES)
    if 0 <= idx < len(hot):
        hot[idx] = 1.0
    return hot


def get_action_embedding(name: str) -> ActionEmbedding:
    """Return the embedding for *name* or a zero vector if missing."""
    vec = _ACTION_VECTORS.get(name, _DEFAULT_VECTOR)
    idx = get_action_index(name)
    return ActionEmbedding(name, idx, list(vec))


def get_embedding_matrix(include_discrete: bool = False) -> List[List[float]] | Any:
    """Return the ordered embedding matrix for all known actions."""

    vectors: Iterable[List[float]] = (
        get_action_embedding(name).as_vector(include_discrete)
        for name in _ACTION_NAMES
    )
    matrix = [list(vector) for vector in vectors]
    xp = get_array_module()
    if xp is not None:
        return xp.asarray(matrix, dtype=xp.float64)
    return matrix
