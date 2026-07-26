"""Role-specific wrappers around the shared unified trunk (see ``unified.py``).

Every class here keeps the exact public name/constructor signature it had in
phase 1, so every existing call site (``Nation._init_ai_controllers``,
``worldsim/military/command.py``, tests, ...) keeps working unchanged — only
the *internals* moved from a per-role trunk (``base.py``/``controller.py``/
``categorical.py``, now removed) onto the single shared trunk in
``unified.py``. The one new constructor parameter every class takes is
``bank`` (a :class:`~worldsim.ai.rnn.unified.UnifiedLoRABank`): omit it for
standalone/test construction (a throwaway bank is created automatically) or
pass the owning nation's shared bank so this role's core LoRA is the same one
every other role for that nation uses (see ``Nation._init_ai_controllers``).
"""
from __future__ import annotations

import random
from collections import deque
from pathlib import Path
from typing import List, Optional

from .unified import (
    BUFFER_SIZE,
    CORE_DIM,
    UnifiedLoRABank,
    _TableRoleView,
)

# Doctrine and fleet-state labels (mirror military.doctrine / military.command;
# duplicated here to avoid a circular import).
_DOCTRINE_LIST = [
    "offensive", "defensive", "economic", "strategic_reserve", "total_war"
]
_FLEET_STATE_LIST = [
    "patrol", "assault", "defend", "escort", "retreat", "reposition", "colonize"
]


def _standalone_bank() -> UnifiedLoRABank:
    """A bank with no real nation behind it — used only when a role wrapper
    is constructed directly (tests, ad-hoc scripts) instead of via
    ``Nation._init_ai_controllers``."""
    return UnifiedLoRABank(core_state_fn=lambda: [0.0] * CORE_DIM)


# -----------------------------------------------------------------------
# Role-specific wrappers
# -----------------------------------------------------------------------

class TorchWarAI(_TableRoleView):
    """Unified-trunk replacement for :class:`~worldsim.ai.WarAI`."""

    def __init__(
        self,
        *,
        table_path:  Optional[str | Path] = None,
        grid_size:   int   = 8,
        allies_dim:  int   = 4,
        epsilon:     float = 0.10,
        gamma:       float = 0.95,
        # kept for API compatibility (ignored — shape is learnt by the GRU)
        hidden_layers: object = None,
        memory_size:   int   = 32,
        bank: Optional[UnifiedLoRABank] = None,
    ) -> None:
        self.grid_size          = max(1, grid_size)
        self.allies_dim         = max(1, allies_dim)
        self.grid_feature_count = self.grid_size * self.grid_size * 2
        n_inputs = 4 + self.allies_dim + self.grid_feature_count
        super().__init__(
            bank or _standalone_bank(), "war", n_inputs, 2,
            table_path=table_path, epsilon=epsilon, gamma=gamma,
        )

    def set_allies_dimension(self, allies_dim: int) -> None:
        """Rebuild the role's extra-feature width when the live nation count changes.

        Only ``extra_proj["war"]`` (shared, not LoRA'd) needs rebuilding —
        the per-nation LoRA lives on the fixed-size core input and the
        fixed-size (n_outputs=2) head, neither of which depends on
        ``allies_dim``, so nothing else needs to change.
        """
        allies_dim = max(1, allies_dim)
        if allies_dim == self.allies_dim:
            return
        self.allies_dim = allies_dim
        new_n = 4 + self.allies_dim + self.grid_feature_count
        self.n_inputs = new_n
        self.trunk.ensure_role("war", new_n, self.n_outputs)
        self._buffer = deque(
            [[0.0] * (CORE_DIM + new_n)] * BUFFER_SIZE, maxlen=BUFFER_SIZE
        )
        self._h = self._h  # unchanged shape (HIDDEN_DIM); nothing to rebuild

    @staticmethod
    def create_doctrine() -> str:
        styles  = ["Blitz", "Trench", "Guerilla", "Mobile", "Integrated"]
        focuses = ["Assault", "Defense", "Strategy", "Doctrine"]
        return f"{random.choice(styles)} {random.choice(focuses)}"


class TorchDomesticPolicyAI(_TableRoleView):
    """Unified-trunk replacement for :class:`~worldsim.ai.DomesticPolicyAI`
    (the legacy flat civilian model — unused when the hierarchical
    :class:`~worldsim.nations.civilian.CivilianController` is active)."""

    def __init__(
        self,
        actions:       int   = 4,
        n_inputs:      int   = 5,
        *,
        table_path:    Optional[str | Path] = None,
        epsilon:       float = 0.10,
        gamma:         float = 0.95,
        hidden_layers: object = None,   # API compat; ignored
        bank: Optional[UnifiedLoRABank] = None,
    ) -> None:
        super().__init__(
            bank or _standalone_bank(), "civilian", n_inputs, actions,
            table_path=table_path, epsilon=epsilon, gamma=gamma,
        )


class TorchCivilianOverseer(_TableRoleView):
    """Overseer model that routes to one of the civilian departments.

    Selects among N departments using the 22-feature state vector. Trained
    with the same per-turn reward signal as the department models it routes
    to.
    """

    def __init__(
        self,
        n_depts:   int   = 6,
        n_inputs:  int   = 22,
        *,
        table_path:    Optional[str | Path] = None,
        epsilon:       float = 0.10,
        gamma:         float = 0.95,
        bank: Optional[UnifiedLoRABank] = None,
    ) -> None:
        super().__init__(
            bank or _standalone_bank(), "civilian_overseer", n_inputs, n_depts,
            table_path=table_path, epsilon=epsilon, gamma=gamma,
        )


class TorchDepartmentAI(_TableRoleView):
    """Department-level model handling one civilian department's local actions.

    Each department gets its own head (role key ``"civilian_{slug}"``) on
    the shared trunk, so departments specialise independently while still
    training the same recurrent weights.
    """

    def __init__(
        self,
        slug:      str,
        n_actions: int,
        n_inputs:  int   = 22,
        *,
        table_path:    Optional[str | Path] = None,
        epsilon:       float = 0.10,
        gamma:         float = 0.95,
        bank: Optional[UnifiedLoRABank] = None,
    ) -> None:
        super().__init__(
            bank or _standalone_bank(), f"civilian_{slug}", n_inputs, n_actions,
            table_path=table_path, epsilon=epsilon, gamma=gamma,
        )
        self.slug = slug


class TorchProjectAI(_TableRoleView):
    """Unified-trunk replacement for :class:`~worldsim.ai.ProjectAI`.

    Takes the full 22-feature civilian state as its extra input (previously
    this role declared ``n_inputs=6`` while actually being fed the 22-feature
    vector, silently truncated — fixed here to genuinely use all of it).
    """

    def __init__(
        self,
        actions:       int,
        n_inputs:      int   = 22,
        *,
        table_path:    Optional[str | Path] = None,
        epsilon:       float = 0.10,
        gamma:         float = 0.95,
        hidden_layers: object = None,
        bank: Optional[UnifiedLoRABank] = None,
    ) -> None:
        super().__init__(
            bank or _standalone_bank(), "project", n_inputs, actions,
            table_path=table_path, epsilon=epsilon, gamma=gamma,
        )


class TorchDiplomacyAI(_TableRoleView):
    """Unified-trunk replacement for :class:`~worldsim.ai.DiplomacyAI`."""

    def __init__(
        self,
        actions:       int   = 3,
        n_inputs:      int   = 5,
        *,
        table_path:    Optional[str | Path] = None,
        epsilon:       float = 0.10,
        gamma:         float = 0.90,
        hidden_layers: object = None,
        bank: Optional[UnifiedLoRABank] = None,
    ) -> None:
        super().__init__(
            bank or _standalone_bank(), "diplomacy", n_inputs, actions,
            table_path=table_path, epsilon=epsilon, gamma=gamma,
        )


class TorchResearchSubsystemAI(_TableRoleView):
    """Unified-trunk backend for one :class:`~worldsim.research.subsystems.TechnologySubsystem`.

    Replaces that subsystem's legacy (non-torch) ``ResearchAI`` — physics,
    engineering and biology each get their own head (role key
    ``"research_{suffix}"``) on the shared trunk.
    """

    def __init__(
        self,
        suffix:    str,
        n_actions: int,
        n_inputs:  int,
        *,
        table_path: Optional[str | Path] = None,
        epsilon:    float = 0.10,
        gamma:      float = 0.90,
        bank: Optional[UnifiedLoRABank] = None,
    ) -> None:
        super().__init__(
            bank or _standalone_bank(), f"research_{suffix}", n_inputs, n_actions,
            table_path=table_path, epsilon=epsilon, gamma=gamma,
        )
        self.n_actions = n_actions   # legacy ResearchAI compatibility attribute


class TorchDoctrineAI(_TableRoleView):
    """Unified-trunk replacement for
    :class:`~worldsim.military.doctrine.DoctrineAI`.

    Provides the same ``choose_doctrine`` / ``train`` interface used by
    :func:`~worldsim.military.command.issue_doctrine`.
    """

    _N_INPUTS  = 10
    _N_OUTPUTS = len(_DOCTRINE_LIST)  # 5

    def __init__(
        self,
        *,
        table_path:    Optional[str | Path] = None,
        epsilon:       float = 0.15,
        gamma:         float = 0.90,
        hidden_layers: object = None,
        bank: Optional[UnifiedLoRABank] = None,
    ) -> None:
        super().__init__(
            bank or _standalone_bank(), "doctrine", self._N_INPUTS, self._N_OUTPUTS,
            table_path=table_path, epsilon=epsilon, gamma=gamma,
        )
        # Compatibility attribute read by DoctrineAI users
        self._optimise_every = 64

    def choose_doctrine(self, state: List[float]) -> str:
        """Return a DoctrineSignal value string for *state*."""
        idx = self.choose_action(state)
        idx = max(0, min(idx, len(_DOCTRINE_LIST) - 1))
        return _DOCTRINE_LIST[idx]

    @staticmethod
    def build_state(nation: object, nations: object) -> List[float]:
        """Delegate to DoctrineAI.build_state (same feature extraction)."""
        from ...military.doctrine import DoctrineAI as _DoctrineAI
        return _DoctrineAI.build_state(nation, nations)  # type: ignore[arg-type]


class TorchEventAI(_TableRoleView):
    """Shared categorical (C51) Q-learning agent for the event system.

    Replaces the per-event-name :class:`~worldsim.events.qlearner.EventQLearner`
    Q-tables with one shared distributional head on the unified trunk,
    LoRA-specialised per nation exactly like every other role. ``n_inputs``
    matches :meth:`~worldsim.events.engine.EventDecisionEngine._state_snapshot`
    (economy, technology.overall, military, infrastructure, stability);
    ``n_outputs`` is a fixed choice-slot cap — events with fewer options mask
    the unused slots via ``choose_action``'s ``valid_mask``.
    """

    MAX_CHOICES: int = 4

    def __init__(
        self,
        *,
        n_inputs:   int   = 5,
        table_path: Optional[str | Path] = None,
        epsilon:    float = 0.10,
        gamma:      float = 0.95,
        bank: Optional[UnifiedLoRABank] = None,
    ) -> None:
        super().__init__(
            bank or _standalone_bank(), "events", n_inputs, self.MAX_CHOICES,
            categorical=True,
            table_path=table_path, epsilon=epsilon, gamma=gamma,
        )


class TorchFleetController(_TableRoleView):
    """Unified-trunk replacement for
    :class:`~worldsim.military.command.LayeredNetworkFleetController`.
    """

    EPSILON: float = 0.12  # kept for FSM compatibility check

    def __init__(
        self,
        n_inputs:  int = 15,
        n_outputs: int = 7,  # len(FLEET_STATE_LIST)
        *,
        table_path: Optional[str | Path] = None,
        bank: Optional[UnifiedLoRABank] = None,
    ) -> None:
        super().__init__(
            bank or _standalone_bank(), "fleet", n_inputs, n_outputs,
            table_path=table_path,
            epsilon=self.EPSILON,
            gamma=0.92,
        )
