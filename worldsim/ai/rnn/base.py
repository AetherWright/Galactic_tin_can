"""Shared GRU backbones, the role registry and replay-buffer training.

One :class:`RoleBaseModel` exists per AI role (war, civilian, project,
diplomacy, research, doctrine, fleet).  All nations sharing a role deposit
experience here; the main simulation thread calls
:func:`step_all_base_models` once per century to train the backbone.

This module assumes PyTorch is importable — :mod:`worldsim.ai.rnn` only
imports it after a successful ``import torch`` probe.
"""
from __future__ import annotations

import random
import threading
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ...core.memory import auto_batch_size

# Prefer CUDA when available; otherwise fall back to CPU.
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Number of past input states kept in the rolling buffer.
BUFFER_SIZE: int = 5

# Doctrine and fleet-state labels (mirror military.doctrine to avoid circular import)
_DOCTRINE_LIST: List[str] = [
    "offensive", "defensive", "economic", "strategic_reserve", "total_war"
]
_FLEET_STATE_LIST: List[str] = [
    "patrol", "assault", "defend", "escort", "retreat", "reposition", "colonize"
]


# -----------------------------------------------------------------------
# Shared GRU backbone
# -----------------------------------------------------------------------

class _BaseGRUModel(nn.Module):
    """Shared GRU backbone for one AI role.

    Layers
    ------
    input_proj  : Linear(input_dim  → hidden_dim)
    gru         : GRU  (hidden_dim  → hidden_dim, batch_first=True)
    output_proj : Linear(hidden_dim → n_outputs)

    LoRA adapters in :class:`RNNController` wrap *input_proj* and
    *output_proj* at forward time without modifying the stored weights.
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, n_outputs: int
    ) -> None:
        super().__init__()
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.n_outputs  = n_outputs
        self.input_proj  = nn.Linear(input_dim,  hidden_dim)
        self.gru         = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.output_proj = nn.Linear(hidden_dim, n_outputs)
        self.to(_DEVICE)

    def forward(
        self,
        x_seq: "torch.Tensor",   # [1, T, input_dim]
        h: "torch.Tensor",        # [1, 1, hidden_dim]
    ) -> "torch.Tensor":           # [1, n_outputs]
        """Base forward without any LoRA correction."""
        T = x_seq.shape[1]
        x_flat = x_seq.reshape(T, self.input_dim)
        x_proj = self.input_proj(x_flat).unsqueeze(0)   # [1, T, hidden_dim]
        gru_out, _ = self.gru(x_proj, h)
        return self.output_proj(gru_out[:, -1])           # [1, n_outputs]

# -----------------------------------------------------------------------
# Per-role shared model + replay buffer
# -----------------------------------------------------------------------

class RoleBaseModel:
    """Shared backbone + replay buffer for a single AI role.

    One instance per role name — retrieved via :func:`get_role_model`.
    All nations sharing this role deposit experiences here; the main
    thread calls :meth:`update_base` (via :func:`step_all_base_models`)
    to run gradient updates on the backbone.
    """

    _BASE_LR:    float = 1e-4
    #: Ceiling for the replay buffer; the actual capacity is derived from the
    #: host RAM budget at construction time (a war-role experience is ~40 KB
    #: of nested Python lists, so 8 000 of them is ~300 MB per role model).
    _REPLAY_CAP: int   = 8_000
    #: Ceiling for the per-step training batch; the actual size is derived
    #: from the free VRAM (or RAM on CPU) at each update.
    _BATCH_SIZE: int   = 64
    _GAMMA:      float = 0.95

    def __init__(
        self, input_dim: int, hidden_dim: int, n_outputs: int
    ) -> None:
        self.model     = _BaseGRUModel(input_dim, hidden_dim, n_outputs)
        self._optimizer = optim.Adam(self.model.parameters(), lr=self._BASE_LR)
        # Lock serialises backward passes from concurrent nation threads.
        self._lock     = threading.Lock()
        # Replay buffer: each entry is (cur_buf, action, reward, next_buf).
        # Experiences are nested Python float lists: ~32 B per value plus list
        # overhead, two BUFFER_SIZE×input_dim sequences per entry.
        replay_item_nbytes = 2 * BUFFER_SIZE * max(1, input_dim) * 32 + 700
        replay_cap = auto_batch_size(
            replay_item_nbytes,
            device="ram",
            fraction=0.05,
            lo=1_000,
            hi=self._REPLAY_CAP,
            default=self._REPLAY_CAP,
        )
        self._replay: Deque[
            Tuple[List[List[float]], int, float, List[List[float]]]
        ] = deque(maxlen=replay_cap)
        self._replay_lock = threading.Lock()

    # ------------------------------------------------------------------

    def add_experience(
        self,
        exp: Tuple[List[List[float]], int, float, List[List[float]]],
    ) -> None:
        with self._replay_lock:
            self._replay.append(exp)

    def update_base(self, n_steps: int = 4) -> None:
        """Run *n_steps* SGD updates on the base model from the replay buffer.

        Call this from the main simulation thread once per century.
        """
        with self._replay_lock:
            replay_snap = list(self._replay)

        # Budget the batch against the memory the forward/backward pass will
        # actually live in: two float32 sequences per sample plus GRU
        # activations and gradient/optimizer headroom (~4x).
        sample_nbytes = (
            2 * BUFFER_SIZE * self.model.input_dim + 12 * self.model.hidden_dim
        ) * 4 * 4
        batch_size = auto_batch_size(
            sample_nbytes,
            device="vram" if _DEVICE.type == "cuda" else "ram",
            fraction=0.10,
            lo=16,
            hi=256,
            default=self._BATCH_SIZE,
        )
        # Keep the original warm-up point: training starts once the buffer
        # holds a legacy-sized batch even when the budget allows bigger ones.
        if len(replay_snap) < min(self._BATCH_SIZE, batch_size):
            return

        h0 = torch.zeros(1, 1, self.model.hidden_dim, device=_DEVICE)

        for _ in range(n_steps):
            batch = random.sample(
                replay_snap, min(batch_size, len(replay_snap))
            )
            total_loss = torch.zeros(1, device=_DEVICE)
            for cur_buf, action, reward, next_buf in batch:
                cur_seq  = torch.tensor(cur_buf,  dtype=torch.float32, device=_DEVICE).unsqueeze(0)
                next_seq = torch.tensor(next_buf, dtype=torch.float32, device=_DEVICE).unsqueeze(0)

                with torch.no_grad():
                    next_q = self.model(next_seq, h0)
                    tv = reward + self._GAMMA * float(next_q.max())

                cur_q  = self.model(cur_seq, h0)
                target = cur_q.detach().clone()
                if 0 <= action < target.shape[1]:
                    target[0, action] = tv
                total_loss = total_loss + F.mse_loss(cur_q, target)

            total_loss = total_loss / len(batch)
            self._optimizer.zero_grad()
            total_loss.backward()
            self._optimizer.step()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path: Path) -> None:
        if path.exists():
            sd = torch.load(path, map_location=_DEVICE, weights_only=True)
            self.model.load_state_dict(sd, strict=False)

# -----------------------------------------------------------------------
# Global role registry
# -----------------------------------------------------------------------

# key: "{role}_{n_inputs}_{n_outputs}"  → RoleBaseModel
_REGISTRY:      Dict[str, RoleBaseModel] = {}
_REGISTRY_LOCK = threading.Lock()


def get_role_model(
    role:       str,
    input_dim:  int,
    hidden_dim: int,
    n_outputs:  int,
) -> RoleBaseModel:
    """Return (creating if absent) the shared base model for *role*."""
    key = f"{role}_{input_dim}_{n_outputs}"
    with _REGISTRY_LOCK:
        if key not in _REGISTRY:
            _REGISTRY[key] = RoleBaseModel(input_dim, hidden_dim, n_outputs)
        return _REGISTRY[key]


def step_all_base_models(n_steps: int = 4) -> None:
    """Update all role base models from their replay buffers.

    Call this from the main simulation thread (not during threaded nation
    turns) once per century.
    """
    for model in list(_REGISTRY.values()):
        model.update_base(n_steps=n_steps)


def save_all_base_models(directory: Path) -> None:
    """Persist every registered role base model to *directory*."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for key, model in _REGISTRY.items():
        model.save(directory / f"base_{key}.pt")


def load_all_base_models(directory: Path) -> None:
    """Restore previously persisted base models from *directory*."""
    directory = Path(directory)
    for key, model in _REGISTRY.items():
        p = directory / f"base_{key}.pt"
        if p.exists():
            model.load(p)
