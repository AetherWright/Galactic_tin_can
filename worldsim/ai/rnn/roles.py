"""Role-specific RNN controller wrappers (Torch counterparts of ai.roles)."""
from __future__ import annotations

import math
import random
from collections import deque
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from .base import BUFFER_SIZE, _DEVICE, _DOCTRINE_LIST, get_role_model
from .controller import _RoleController


# -----------------------------------------------------------------------
# Role-specific wrappers
# -----------------------------------------------------------------------

class TorchWarAI(_RoleController):
    """RNN+LoRA replacement for :class:`~worldsim.ai.WarAI`."""

    def __init__(
        self,
        *,
        table_path:  Optional[str | Path] = None,
        grid_size:   int   = 8,
        allies_dim:  int   = 4,
        epsilon:     float = 0.10,
        gamma:       float = 0.95,
        # kept for API compatibility (ignored — shape is learnt by RNN)
        hidden_layers: object = None,
        memory_size:   int   = 32,
    ) -> None:
        self.grid_size           = max(1, grid_size)
        self.allies_dim          = max(1, allies_dim)
        self.grid_feature_count  = self.grid_size * self.grid_size * 2
        n_inputs = 4 + self.allies_dim + self.grid_feature_count
        super().__init__(
            "war", n_inputs, 2, 128,
            table_path=table_path, epsilon=epsilon, gamma=gamma,
        )

    def set_allies_dimension(self, allies_dim: int) -> None:
        """Rebuild the controller when the total-nation count changes."""
        allies_dim = max(1, allies_dim)
        if allies_dim == self.allies_dim:
            return
        self.allies_dim = allies_dim
        new_n = 4 + self.allies_dim + self.grid_feature_count
        self.n_inputs = new_n
        # Update the shared base model reference (new key if dim changed)
        self.base = get_role_model("war", new_n, self.hidden_dim, self.n_outputs)
        # Re-create LoRA parameters for the new input dimension
        self.lora_A_in  = nn.Parameter(torch.empty(self.lora_r, new_n, device=_DEVICE))
        self.lora_B_in  = nn.Parameter(torch.zeros(self.hidden_dim, self.lora_r, device=_DEVICE))
        nn.init.kaiming_uniform_(self.lora_A_in, a=math.sqrt(5))
        # Rebuild the frozen target tensors so they match the new input
        # dimension.  Without this they keep the old shape and the next
        # Double-DQN forward / target-network sync raises a shape mismatch —
        # which is why the war controller never trained or saved cleanly once
        # the live nation count changed.
        with torch.no_grad():
            self._target_A_in = self.lora_A_in.data.clone()
            self._target_B_in = self.lora_B_in.data.clone()
        self._lora_optimizer = optim.Adam(
            [self.lora_A_in, self.lora_B_in,
             self.lora_A_out, self.lora_B_out],
            lr=self._lr_lora,
        )
        self._buffer = deque([[0.0] * new_n] * BUFFER_SIZE, maxlen=BUFFER_SIZE)
        self._h = torch.zeros(1, 1, self.hidden_dim, device=_DEVICE)

    def load_lora(self, path: Path) -> None:
        """Restore weights, then realign ``allies_dim`` with the loaded size.

        A checkpoint may have been written at a different nation count; the
        base loader adopts the saved input dimension, so keep ``allies_dim``
        in step to avoid a spurious rebuild on the next
        :meth:`set_allies_dimension` call (which would discard the weights).
        """
        super().load_lora(path)
        self.allies_dim = max(1, self.n_inputs - 4 - self.grid_feature_count)

    @staticmethod
    def create_doctrine() -> str:
        styles  = ["Blitz", "Trench", "Guerilla", "Mobile", "Integrated"]
        focuses = ["Assault", "Defense", "Strategy", "Doctrine"]
        return f"{random.choice(styles)} {random.choice(focuses)}"


class TorchDomesticPolicyAI(_RoleController):
    """RNN+LoRA replacement for :class:`~worldsim.ai.DomesticPolicyAI`."""

    def __init__(
        self,
        actions:       int   = 4,
        n_inputs:      int   = 5,
        *,
        table_path:    Optional[str | Path] = None,
        epsilon:       float = 0.10,
        gamma:         float = 0.95,
        hidden_layers: object = None,   # API compat; ignored
    ) -> None:
        super().__init__(
            "civilian", n_inputs, actions, 128,
            table_path=table_path, epsilon=epsilon, gamma=gamma,
        )


class TorchCivilianOverseer(_RoleController):
    """Overseer model that routes to one of the civilian departments.

    Selects among N departments (6-way choice) using the 22-feature state
    vector (20 base features + food_ratio + farm count).  Trained with the
    same per-turn reward signal as the department models it routes to.
    """

    def __init__(
        self,
        n_depts:   int   = 6,
        n_inputs:  int   = 22,
        *,
        table_path:    Optional[str | Path] = None,
        epsilon:       float = 0.10,
        gamma:         float = 0.95,
    ) -> None:
        super().__init__(
            "civilian_overseer", n_inputs, n_depts, 64,
            table_path=table_path, epsilon=epsilon, gamma=gamma,
            dynamic_lr=True,
        )


class TorchDepartmentAI(_RoleController):
    """Department-level model handling one civilian department's local actions.

    Each department gets its own shared base model (role key
    ``"civilian_{slug}"``), so departments specialise independently while
    nations within a department still share the same backbone.
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
    ) -> None:
        super().__init__(
            f"civilian_{slug}", n_inputs, n_actions, 64,
            table_path=table_path, epsilon=epsilon, gamma=gamma,
            dynamic_lr=True,
        )
        self.slug = slug


class TorchProjectAI(_RoleController):
    """RNN+LoRA replacement for :class:`~worldsim.ai.ProjectAI`."""

    def __init__(
        self,
        actions:       int,
        n_inputs:      int   = 6,
        *,
        table_path:    Optional[str | Path] = None,
        epsilon:       float = 0.10,
        gamma:         float = 0.95,
        hidden_layers: object = None,
    ) -> None:
        super().__init__(
            "project", n_inputs, actions, 64,
            table_path=table_path, epsilon=epsilon, gamma=gamma,
        )


class TorchDiplomacyAI(_RoleController):
    """RNN+LoRA replacement for :class:`~worldsim.ai.DiplomacyAI`."""

    def __init__(
        self,
        actions:       int   = 3,
        n_inputs:      int   = 5,
        *,
        table_path:    Optional[str | Path] = None,
        epsilon:       float = 0.10,
        gamma:         float = 0.90,
        hidden_layers: object = None,
    ) -> None:
        super().__init__(
            "diplomacy", n_inputs, actions, 32,
            table_path=table_path, epsilon=epsilon, gamma=gamma,
        )


class TorchResearchAI(_RoleController):
    """RNN+LoRA replacement for :class:`~worldsim.ai.ResearchAI`."""

    def __init__(
        self,
        actions:       int,
        n_inputs:      int   = 4,
        *,
        table_path:    Optional[str | Path] = None,
        epsilon:       float = 0.10,
        gamma:         float = 0.90,
        hidden_layers: object = None,
    ) -> None:
        super().__init__(
            "research", n_inputs, actions, 64,
            table_path=table_path, epsilon=epsilon, gamma=gamma,
        )


class TorchDoctrineAI(_RoleController):
    """RNN+LoRA replacement for
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
    ) -> None:
        super().__init__(
            "doctrine", self._N_INPUTS, self._N_OUTPUTS, 32,
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


class TorchFleetController(_RoleController):
    """RNN+LoRA replacement for
    :class:`~worldsim.military.command.LayeredNetworkFleetController`.
    """

    EPSILON: float = 0.12  # kept for FSM compatibility check

    def __init__(
        self,
        n_inputs:  int = 15,
        n_outputs: int = 7,  # len(FLEET_STATE_LIST)
        *,
        table_path: Optional[str | Path] = None,
    ) -> None:
        super().__init__(
            "fleet", n_inputs, n_outputs, 32,
            table_path=table_path,
            epsilon=self.EPSILON,
            gamma=0.92,
        )
