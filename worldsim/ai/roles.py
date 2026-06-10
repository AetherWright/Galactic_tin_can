"""Role-specific facades over :class:`~worldsim.ai.policy.NelderMeadPolicy`."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence
import random

from .networks import LayeredNetwork
from .policy import NelderMeadPolicy


class WarAI(NelderMeadPolicy):
    """Layered Nelder-Mead controller for the military AI."""

    def __init__(
        self,
        *,
        table_path: str | Path | None = None,
        grid_size: int = 8,
        allies_dim: int = 4,
        epsilon: float = 0.1,
        gamma: float = 0.95,
        hidden_layers: Sequence[int] = (20, 16, 12, 9, 6, 4, 6, 9, 12, 16, 20),
        memory_size: int = 32,
    ) -> None:
        self.grid_size = max(1, grid_size)
        self.allies_dim = max(1, allies_dim)
        self.grid_feature_count = self.grid_size * self.grid_size * 2
        n_inputs = 4 + self.allies_dim + self.grid_feature_count
        super().__init__(
            n_inputs,
            2,
            table_path=table_path,
            epsilon=epsilon,
            gamma=gamma,
            hidden_layers=hidden_layers,
            memory_size=memory_size,
        )

    def set_allies_dimension(self, allies_dim: int) -> None:
        """Adjust the expected size of the ally feature vector."""

        allies_dim = max(1, allies_dim)
        if allies_dim == self.allies_dim:
            return
        self.allies_dim = allies_dim
        self.n_inputs = 4 + self.allies_dim + self.grid_feature_count
        self.network = LayeredNetwork(
            self.n_inputs,
            [*self.hidden_layers, self.n_actions],
            seed=self._rng.randint(0, 2**31 - 1),
        )
        self.scale_count = self._expected_scale_length()
        self.scales = [1.0] * self.scale_count
        self.memory.clear()

    def create_doctrine(self) -> str:
        styles = ["Blitz", "Trench", "Guerilla", "Mobile", "Integrated"]
        focuses = ["Assault", "Defense", "Strategy", "Doctrine"]
        return f"{random.choice(styles)} {random.choice(focuses)}"


class DomesticPolicyAI(NelderMeadPolicy):
    """Nelder-Mead based learner for domestic policy."""

    def __init__(
        self,
        actions: int = 4,
        n_inputs: int = 5,
        *,
        table_path: str | Path | None = None,
        epsilon: float = 0.1,
        gamma: float = 0.95,
        hidden_layers: Sequence[int] = (12, 9, 7, 5, 4, 5, 7, 9, 12),
    ) -> None:
        super().__init__(
            n_inputs,
            actions,
            table_path=table_path,
            epsilon=epsilon,
            gamma=gamma,
            hidden_layers=hidden_layers,
        )


class CivilianOverseerAI(NelderMeadPolicy):
    """Nelder-Mead overseer: routes to one of the N civilian departments."""

    def __init__(
        self,
        n_depts: int = 6,
        n_inputs: int = 22,
        *,
        table_path: str | Path | None = None,
        epsilon: float = 0.10,
        gamma: float = 0.95,
        hidden_layers: Sequence[int] = (16, 12, 8, 6, 8, 12, 16),
    ) -> None:
        super().__init__(
            n_inputs, n_depts,
            table_path=table_path, epsilon=epsilon, gamma=gamma,
            hidden_layers=hidden_layers,
        )


class DepartmentPolicyAI(NelderMeadPolicy):
    """Nelder-Mead learner for a single department's local action sub-space."""

    def __init__(
        self,
        slug: str,
        n_actions: int,
        n_inputs: int = 22,
        *,
        table_path: str | Path | None = None,
        epsilon: float = 0.10,
        gamma: float = 0.95,
        hidden_layers: Sequence[int] = (12, 9, 7, 5, 7, 9, 12),
    ) -> None:
        super().__init__(
            n_inputs, n_actions,
            table_path=table_path, epsilon=epsilon, gamma=gamma,
            hidden_layers=hidden_layers,
        )
        self.slug = slug


class ProjectAI(NelderMeadPolicy):
    """Nelder-Mead learner for construction project decisions."""

    def __init__(
        self,
        actions: int,
        n_inputs: int = 6,
        *,
        table_path: str | Path | None = None,
        epsilon: float = 0.1,
        gamma: float = 0.95,
        hidden_layers: Sequence[int] = (14, 11, 9, 7, 5, 4, 5, 7 ,9, 11, 14),
    ) -> None:
        super().__init__(
            n_inputs,
            actions,
            table_path=table_path,
            epsilon=epsilon,
            gamma=gamma,
            hidden_layers=hidden_layers,
        )


class DiplomacyAI(NelderMeadPolicy):
    """Nelder-Mead learner for diplomacy choices."""

    def __init__(
        self,
        actions: int = 3,
        n_inputs: int = 5,
        *,
        table_path: str | Path | None = None,
        epsilon: float = 0.1,
        gamma: float = 0.9,
        hidden_layers: Sequence[int] = (10, 8, 6, 4, 3, 4, 6, 8, 10),
    ) -> None:
        super().__init__(
            n_inputs,
            actions,
            table_path=table_path,
            epsilon=epsilon,
            gamma=gamma,
            hidden_layers=hidden_layers,
        )


class ResearchAI(NelderMeadPolicy):
    """Nelder-Mead learner for technology selection."""

    def __init__(
        self,
        actions: int,
        n_inputs: int = 4,
        *,
        table_path: str | Path | None = None,
        epsilon: float = 0.1,
        gamma: float = 0.9,
        hidden_layers: Sequence[int] = (10, 8, 6, 4, 3, 4, 6, 8, 10),
    ) -> None:
        super().__init__(
            n_inputs,
            actions,
            table_path=table_path,
            epsilon=epsilon,
            gamma=gamma,
            hidden_layers=hidden_layers,
        )

