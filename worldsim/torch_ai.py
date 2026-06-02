"""Optional PyTorch-backed fleet controller for WorldSim.

:class:`TorchFleetController` is set to ``None`` when ``torch`` is not
installed so the rest of the codebase can check ``if TorchFleetController``
before using it.  The pure-Python :class:`LayeredNetworkFleetController` in
:mod:`worldsim.models.military_ai` is always available as a fallback.

Interface
---------
Both controllers share the same two-method surface:

``predict(inputs: List[float]) -> List[float]``
    Forward pass; returns one score per FSM state.

``train(state, action, reward, next_state)``
    Online TD(0) update for the chosen action.
"""
from __future__ import annotations

from typing import List

_TORCH_AVAILABLE: bool = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    _TORCH_AVAILABLE = True
except ImportError:
    pass

# Prefer CUDA when available; otherwise fall back to CPU.
_DEVICE: "torch.device"
if _TORCH_AVAILABLE:
    import torch as _torch_for_device
    _DEVICE = _torch_for_device.device("cuda" if _torch_for_device.cuda.is_available() else "cpu")
    del _torch_for_device


def torch_available() -> bool:
    """Return ``True`` if ``torch`` is importable."""
    return _TORCH_AVAILABLE


# ---------------------------------------------------------------------------
# PyTorch implementation (only defined when torch is present)
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    class _MLP(nn.Module):
        """Three-layer perceptron used inside :class:`TorchFleetController`."""

        def __init__(
            self,
            n_in: int,
            n_out: int,
            hidden: tuple = (32, 24),
        ) -> None:
            super().__init__()
            layers: list = []
            prev = n_in
            for h in hidden:
                layers += [nn.Linear(prev, h), nn.ReLU()]
                prev = h
            layers.append(nn.Linear(prev, n_out))
            self.net = nn.Sequential(*layers)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.net(x)

    class TorchFleetController:
        """Fleet neural controller backed by a PyTorch MLP.

        Training uses online TD(0): after each decision the target Q-value is
        ``reward + γ * max(Q(next_state))``, and a single gradient step is
        taken via Adam.

        Parameters
        ----------
        n_inputs:
            Feature-vector length (default 15; see
            :func:`~worldsim.models.military_ai.build_fleet_features`).
        n_outputs:
            Number of FSM states to score (default 6).
        hidden:
            Hidden layer widths for the MLP.
        lr:
            Adam learning rate.
        gamma:
            TD discount factor.
        """

        def __init__(
            self,
            n_inputs: int = 15,
            n_outputs: int = 6,
            *,
            hidden: tuple = (32, 24),
            lr: float = 1e-3,
            gamma: float = 0.92,
        ) -> None:
            self.n_inputs = n_inputs
            self.n_outputs = n_outputs
            self.gamma = gamma
            self.model = _MLP(n_inputs, n_outputs, hidden).to(_DEVICE)
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
            self.loss_fn = nn.MSELoss()

        # ------------------------------------------------------------------
        # Helpers
        # ------------------------------------------------------------------

        def _to_tensor(self, inputs: List[float]) -> "torch.Tensor":
            vec = list(inputs)[: self.n_inputs]
            vec += [0.0] * max(0, self.n_inputs - len(vec))
            return torch.tensor([vec], dtype=torch.float32, device=_DEVICE)

        # ------------------------------------------------------------------
        # Public API
        # ------------------------------------------------------------------

        def predict(self, inputs: List[float]) -> List[float]:
            """Return one score per FSM state (no gradient)."""
            with torch.no_grad():
                out = self.model(self._to_tensor(inputs))
            return out[0].tolist()

        def train(
            self,
            state: List[float],
            action: int,
            reward: float,
            next_state: List[float],
        ) -> None:
            """Apply one online TD(0) gradient step."""
            if not (0 <= action < self.n_outputs):
                return

            state_t = self._to_tensor(state)
            next_t = self._to_tensor(next_state)

            with torch.no_grad():
                next_q = self.model(next_t)
                target_val = reward + self.gamma * float(next_q.max())

            current_q = self.model(state_t)
            target = current_q.clone().detach()
            target[0, action] = target_val

            loss = self.loss_fn(current_q, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # ------------------------------------------------------------------
        # Serialization
        # ------------------------------------------------------------------

        def to_dict(self) -> dict:
            return {
                "type": "torch_fleet_controller",
                "n_inputs": self.n_inputs,
                "n_outputs": self.n_outputs,
                "state_dict": {
                    k: v.tolist() for k, v in self.model.state_dict().items()
                },
            }

        @classmethod
        def from_dict(cls, payload: dict) -> "TorchFleetController":
            obj = cls(
                n_inputs=int(payload.get("n_inputs", 15)),
                n_outputs=int(payload.get("n_outputs", 6)),
            )
            sd = payload.get("state_dict")
            if isinstance(sd, dict):
                obj.model.load_state_dict(
                    {k: torch.tensor(v, device=_DEVICE) for k, v in sd.items()},
                    strict=False,
                )
            return obj

else:
    TorchFleetController = None  # type: ignore[assignment,misc]
