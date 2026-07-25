"""Categorical (C51-style) distributional Q-learning — shared events model.

Every other AI role in this package (:mod:`.base` / :mod:`.controller`)
regresses a single scalar Q-value per action with Double DQN + MSE. The
event system instead learns a full return *distribution* per action over a
fixed set of support atoms (Bellemare, Dabney & Munos, "A Distributional
Perspective on Reinforcement Learning", 2017 — "C51").

This replaces what used to be a hand-rolled Q-*table* per event name
(:class:`worldsim.events.qlearner.EventQLearner` — a fresh table, learning
from scratch, for every procedurally-generated event name). One shared
categorical network, conditioned on nation state, generalises across every
event instead, and per-nation specialisation still uses the same LoRA trick
as every other role: the shared trunk trains from a pooled replay buffer,
the per-nation LoRA adapter trains online on every decision.

The shared trunk lives in the same registry as the scalar role models
(:data:`worldsim.ai.rnn.base._REGISTRY`) under a ``"cat_"``-prefixed key, so
:func:`~worldsim.ai.rnn.base.step_all_base_models`,
:func:`~worldsim.ai.rnn.base.save_all_base_models` and
:func:`~worldsim.ai.rnn.base.load_all_base_models` already pick it up for
free — both model classes expose the same ``update_base`` / ``save`` /
``load`` surface.
"""
from __future__ import annotations

import math
import random
import threading
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ...core.memory import auto_batch_size
from .base import BUFFER_SIZE, _DEVICE, _REGISTRY, _REGISTRY_LOCK

# ---------------------------------------------------------------------------
# Fixed atom support
# ---------------------------------------------------------------------------
# Sized to the event system's effect magnitudes (worldsim/events/catalog.py,
# worldsim/events/generator.py effects run roughly +/-40; +/-50 gives
# headroom without wasting resolution).
N_ATOMS: int = 51
V_MIN:   float = -50.0
V_MAX:   float = 50.0
_DELTA_Z: float = (V_MAX - V_MIN) / (N_ATOMS - 1)

_SUPPORT: torch.Tensor = torch.linspace(V_MIN, V_MAX, N_ATOMS, device=_DEVICE)


def project_categorical(
    next_dist: torch.Tensor,   # [B, N_ATOMS] probabilities (not log)
    rewards:   torch.Tensor,   # [B]
    gamma:     float,
    support:   torch.Tensor = _SUPPORT,
) -> torch.Tensor:              # [B, N_ATOMS]
    """Project the categorical Bellman target onto the fixed atom support.

    Standard C51 projection: ``Tz_j = clip(r + gamma * z_j)``, then spread
    each atom's probability mass linearly onto its two neighbouring support
    atoms so the result stays a valid distribution over the same fixed
    support (needed because ``Tz_j`` generally does not land exactly on a
    support atom).
    """
    batch = next_dist.shape[0]
    n_atoms = support.shape[0]
    delta_z = (V_MAX - V_MIN) / (n_atoms - 1)

    tz = (rewards.view(batch, 1) + gamma * support.view(1, n_atoms))
    tz = tz.clamp(V_MIN, V_MAX)
    b = (tz - V_MIN) / delta_z
    lower = b.floor()
    upper = b.ceil()
    same = lower == upper

    # Mass split proportional to distance from each neighbour; when b lands
    # exactly on an atom (lower == upper) all of it goes to that atom.
    upper_weight = torch.where(same, torch.zeros_like(b), upper - b)
    lower_weight = torch.where(same, torch.ones_like(b), b - lower)

    lower_idx = lower.long().clamp(0, n_atoms - 1)
    upper_idx = upper.long().clamp(0, n_atoms - 1)

    projected = torch.zeros(batch, n_atoms, device=next_dist.device)
    projected.scatter_add_(1, lower_idx, next_dist * upper_weight)
    projected.scatter_add_(1, upper_idx, next_dist * lower_weight)
    return projected


# ---------------------------------------------------------------------------
# Shared categorical GRU backbone
# ---------------------------------------------------------------------------

class _CategoricalGRUModel(nn.Module):
    """Shared GRU backbone producing a categorical distribution per action.

    Same ``input_proj -> GRU -> output_proj`` shape as
    :class:`~worldsim.ai.rnn.base._BaseGRUModel`, but ``output_proj`` maps to
    ``n_outputs * N_ATOMS`` logits — one softmax distribution per action
    instead of one scalar per action.
    """

    def __init__(self, input_dim: int, hidden_dim: int, n_outputs: int) -> None:
        super().__init__()
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.n_outputs  = n_outputs
        self.input_proj  = nn.Linear(input_dim, hidden_dim)
        self.gru         = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.output_proj = nn.Linear(hidden_dim, n_outputs * N_ATOMS)
        self.to(_DEVICE)

    def forward(
        self,
        x_seq: "torch.Tensor",   # [1, T, input_dim]
        h:     "torch.Tensor",   # [1, 1, hidden_dim]
    ) -> "torch.Tensor":           # [1, n_outputs, N_ATOMS] log-probs
        """Base forward without any LoRA correction."""
        T = x_seq.shape[1]
        x_flat = x_seq.reshape(T, self.input_dim)
        x_proj = self.input_proj(x_flat).unsqueeze(0)
        gru_out, _ = self.gru(x_proj, h)
        logits = self.output_proj(gru_out[:, -1]).view(-1, self.n_outputs, N_ATOMS)
        return F.log_softmax(logits, dim=-1)


# ---------------------------------------------------------------------------
# Per-role shared model + replay buffer (categorical loss)
# ---------------------------------------------------------------------------

class CategoricalRoleBaseModel:
    """Shared categorical backbone + replay buffer for a single AI role.

    Mirrors :class:`~worldsim.ai.rnn.base.RoleBaseModel` (replay deque,
    memory-budgeted batch size, threading lock) but ``update_base`` trains
    against the projected categorical Bellman target with cross-entropy
    instead of MSE against a scalar target.
    """

    _BASE_LR:    float = 1e-4
    _REPLAY_CAP: int   = 8_000
    _BATCH_SIZE: int   = 64
    _GAMMA:      float = 0.95

    def __init__(self, input_dim: int, hidden_dim: int, n_outputs: int) -> None:
        self.model      = _CategoricalGRUModel(input_dim, hidden_dim, n_outputs)
        self._optimizer = optim.Adam(self.model.parameters(), lr=self._BASE_LR)
        self._lock      = threading.Lock()
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

    def add_experience(
        self,
        exp: Tuple[List[List[float]], int, float, List[List[float]]],
    ) -> None:
        with self._replay_lock:
            self._replay.append(exp)

    def update_base(self, n_steps: int = 4) -> None:
        """Run *n_steps* SGD updates on the base model from the replay buffer."""
        with self._replay_lock:
            replay_snap = list(self._replay)

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
        if len(replay_snap) < min(self._BATCH_SIZE, batch_size):
            return

        h0 = torch.zeros(1, 1, self.model.hidden_dim, device=_DEVICE)

        for _ in range(n_steps):
            batch = random.sample(replay_snap, min(batch_size, len(replay_snap)))
            total_loss = torch.zeros(1, device=_DEVICE)
            for cur_buf, action, reward, next_buf in batch:
                cur_seq  = torch.tensor(cur_buf,  dtype=torch.float32, device=_DEVICE).unsqueeze(0)
                next_seq = torch.tensor(next_buf, dtype=torch.float32, device=_DEVICE).unsqueeze(0)
                reward_t = torch.tensor([reward], dtype=torch.float32, device=_DEVICE)

                with torch.no_grad():
                    next_log_probs = self.model(next_seq, h0)
                    next_probs     = next_log_probs.exp()
                    expected       = (next_probs * _SUPPORT).sum(dim=-1)
                    best_next_act  = int(expected.argmax(dim=1).item())
                    next_dist      = next_probs[:, best_next_act]
                    target_dist    = project_categorical(next_dist, reward_t, self._GAMMA)

                cur_log_probs = self.model(cur_seq, h0)
                if 0 <= action < cur_log_probs.shape[1]:
                    log_p_a = cur_log_probs[0, action]
                    total_loss = total_loss - (target_dist[0] * log_p_a).sum()

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


def get_categorical_role_model(
    role:       str,
    input_dim:  int,
    hidden_dim: int,
    n_outputs:  int,
) -> CategoricalRoleBaseModel:
    """Return (creating if absent) the shared categorical base model for *role*.

    Stored in the same registry as :func:`~worldsim.ai.rnn.base.get_role_model`
    (key-prefixed with ``"cat_"`` so it can't collide) so the existing
    ``step_all_base_models`` / ``save_all_base_models`` / ``load_all_base_models``
    already train and persist it without any extra wiring.
    """
    key = f"cat_{role}_{input_dim}_{n_outputs}"
    with _REGISTRY_LOCK:
        if key not in _REGISTRY:
            _REGISTRY[key] = CategoricalRoleBaseModel(input_dim, hidden_dim, n_outputs)
        return _REGISTRY[key]   # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Per-nation categorical controller with LoRA
# ---------------------------------------------------------------------------

class CategoricalRNNController:
    """Nation-specific categorical GRU controller with LoRA specialisation.

    Same rolling input buffer / per-nation LoRA / Double-DQN-style target
    network as :class:`~worldsim.ai.rnn.controller.RNNController`, but the
    value head is a categorical distribution over :data:`N_ATOMS` fixed
    atoms rather than a scalar, and the training loss is the C51 cross
    entropy against the projected Bellman target instead of MSE.
    """

    _LORA_R:            int   = 8
    _LORA_ALPHA:        float = 16.0
    _GAMMA:             float = 0.95
    _EPSILON:           float = 0.10
    _LR_LORA:           float = 3e-4
    TARGET_UPDATE_FREQ: int   = 16

    def __init__(
        self,
        role:       str,
        n_inputs:   int,
        n_outputs:  int,
        hidden_dim: int = 64,
        *,
        lora_r:     int   = _LORA_R,
        lora_alpha: float = _LORA_ALPHA,
        epsilon:    float = _EPSILON,
        gamma:      float = _GAMMA,
        lr_lora:    float = _LR_LORA,
        seed:       Optional[int] = None,
    ) -> None:
        self.role       = role
        self.n_inputs   = n_inputs
        self.n_outputs  = n_outputs
        self.hidden_dim = hidden_dim
        self.lora_r     = lora_r
        self.lora_alpha = lora_alpha
        self.epsilon    = epsilon
        self.gamma      = gamma
        self._rng       = random.Random(seed)

        self.base = get_categorical_role_model(role, n_inputs, hidden_dim, n_outputs)

        # LoRA parameters — nation-specific (online network)
        self.lora_A_in  = nn.Parameter(torch.empty(lora_r, n_inputs, device=_DEVICE))
        self.lora_B_in  = nn.Parameter(torch.zeros(hidden_dim, lora_r, device=_DEVICE))
        self.lora_A_out = nn.Parameter(torch.empty(lora_r, hidden_dim, device=_DEVICE))
        self.lora_B_out = nn.Parameter(
            torch.zeros(n_outputs * N_ATOMS, lora_r, device=_DEVICE)
        )

        nn.init.kaiming_uniform_(self.lora_A_in,  a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A_out, a=math.sqrt(5))

        self._lora_optimizer = optim.Adam(
            [self.lora_A_in, self.lora_B_in, self.lora_A_out, self.lora_B_out],
            lr=lr_lora,
        )
        self._lr_lora = lr_lora

        with torch.no_grad():
            self._target_A_in  = self.lora_A_in.data.clone()
            self._target_B_in  = self.lora_B_in.data.clone()
            self._target_A_out = self.lora_A_out.data.clone()
            self._target_B_out = self.lora_B_out.data.clone()
        self._train_steps: int = 0

        self._h: torch.Tensor = torch.zeros(1, 1, hidden_dim, device=_DEVICE)
        self._buffer: Deque[List[float]] = deque(
            [[0.0] * n_inputs] * BUFFER_SIZE, maxlen=BUFFER_SIZE
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _pad(self, state: Sequence[float]) -> List[float]:
        vec = list(state)[: self.n_inputs]
        vec += [0.0] * max(0, self.n_inputs - len(vec))
        return vec

    def _forward_dist(
        self, x_seq: "torch.Tensor", h: "torch.Tensor"
    ) -> "torch.Tensor":   # [1, n_outputs, N_ATOMS] log-probs, online LoRA
        base  = self.base.model
        scale = self.lora_alpha / self.lora_r
        T     = x_seq.shape[1]
        x_flat = x_seq.reshape(T, self.n_inputs)

        proj_in  = F.linear(x_flat, base.input_proj.weight, base.input_proj.bias)
        lora_in_ = (x_flat @ self.lora_A_in.T) @ self.lora_B_in.T * scale
        x_proj   = (proj_in + lora_in_).unsqueeze(0)

        gru_out, _ = base.gru(x_proj, h)
        last = gru_out[:, -1]

        proj_out  = F.linear(last, base.output_proj.weight, base.output_proj.bias)
        lora_out_ = (last @ self.lora_A_out.T) @ self.lora_B_out.T * scale
        logits    = (proj_out + lora_out_).view(-1, self.n_outputs, N_ATOMS)
        return F.log_softmax(logits, dim=-1)

    def _forward_dist_target(
        self, x_seq: "torch.Tensor", h: "torch.Tensor"
    ) -> "torch.Tensor":   # frozen target LoRA
        base  = self.base.model
        scale = self.lora_alpha / self.lora_r
        T     = x_seq.shape[1]
        x_flat = x_seq.reshape(T, self.n_inputs)

        proj_in  = F.linear(x_flat, base.input_proj.weight, base.input_proj.bias)
        lora_in_ = (x_flat @ self._target_A_in.T) @ self._target_B_in.T * scale
        x_proj   = (proj_in + lora_in_).unsqueeze(0)

        gru_out, _ = base.gru(x_proj, h)
        last = gru_out[:, -1]

        proj_out  = F.linear(last, base.output_proj.weight, base.output_proj.bias)
        lora_out_ = (last @ self._target_A_out.T) @ self._target_B_out.T * scale
        logits    = (proj_out + lora_out_).view(-1, self.n_outputs, N_ATOMS)
        return F.log_softmax(logits, dim=-1)

    def _update_target_network(self) -> None:
        with torch.no_grad():
            self._target_A_in.copy_(self.lora_A_in.data)
            self._target_B_in.copy_(self.lora_B_in.data)
            self._target_A_out.copy_(self.lora_A_out.data)
            self._target_B_out.copy_(self.lora_B_out.data)

    @staticmethod
    def _buf_to_tensor(buf: Deque) -> "torch.Tensor":
        return torch.tensor(list(buf), dtype=torch.float32, device=_DEVICE).unsqueeze(0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, state: Sequence[float]) -> List[float]:
        """Forward pass; returns the expected value (Σ atom·P(atom)) per action."""
        vec = self._pad(state)
        self._buffer.append(vec)
        seq = self._buf_to_tensor(self._buffer)
        with torch.no_grad():
            log_probs = self._forward_dist(seq, self._h)
            expected  = (log_probs.exp() * _SUPPORT).sum(dim=-1)
            x_flat = seq.reshape(seq.shape[1], self.n_inputs)
            proj = F.linear(
                x_flat, self.base.model.input_proj.weight, self.base.model.input_proj.bias
            )
            _, h_new = self.base.model.gru(proj.unsqueeze(0), self._h)
        self._h = h_new.detach()
        return expected[0].tolist()

    def choose_action(
        self,
        state:      Sequence[float],
        valid_mask: Optional[List[bool]] = None,
    ) -> int:
        """ε-greedy action selection over the expected values."""
        if self.n_outputs <= 0:
            return 0
        scores = self.predict(state)
        n = min(self.n_outputs, len(scores))
        if n == 0:
            return 0
        if self._rng.random() < self.epsilon:
            if valid_mask:
                valid = [i for i in range(n) if valid_mask[i]]
                return self._rng.choice(valid) if valid else 0
            return self._rng.randrange(n)
        best_idx, best_val = 0, float("-inf")
        for i in range(n):
            if valid_mask and not valid_mask[i]:
                continue
            if scores[i] > best_val:
                best_idx, best_val = i, scores[i]
        return best_idx

    def train(
        self,
        state:      Sequence[float],
        action:     int,
        reward:     float,
        next_state: Sequence[float],
    ) -> None:
        """C51 Double-DQN-style update: cross-entropy against the projected
        categorical Bellman target, computed the same way
        :class:`~worldsim.ai.rnn.controller.RNNController` computes its
        scalar Double DQN target (online net selects the next action, target
        net evaluates/produces the distribution being bootstrapped from).
        """
        if not (0 <= action < self.n_outputs):
            return

        next_vec  = self._pad(next_state)
        cur_buf   = list(self._buffer)
        next_buf_deq: Deque[List[float]] = deque(self._buffer, maxlen=BUFFER_SIZE)
        next_buf_deq.append(next_vec)
        next_buf = list(next_buf_deq)

        cur_seq  = torch.tensor(cur_buf,  dtype=torch.float32, device=_DEVICE).unsqueeze(0)
        next_seq = torch.tensor(next_buf, dtype=torch.float32, device=_DEVICE).unsqueeze(0)
        reward_t = torch.tensor([reward], dtype=torch.float32, device=_DEVICE)
        h_ref    = self._h

        with self.base._lock:
            with torch.no_grad():
                next_log_probs_online = self._forward_dist(next_seq, h_ref)
                next_probs_online     = next_log_probs_online.exp()
                expected_online        = (next_probs_online * _SUPPORT).sum(dim=-1)
                best_next_act          = int(expected_online.argmax(dim=1).item())

                next_log_probs_target = self._forward_dist_target(next_seq, h_ref)
                next_dist_target      = next_log_probs_target.exp()[:, best_next_act]
                target_dist           = project_categorical(
                    next_dist_target, reward_t, self.gamma
                )

            cur_log_probs = self._forward_dist(cur_seq, h_ref)
            log_p_a = cur_log_probs[0, action]
            loss = -(target_dist[0] * log_p_a).sum()

            self._lora_optimizer.zero_grad()
            loss.backward()
            self._lora_optimizer.step()

            self.base.model.zero_grad()

        self._train_steps += 1
        if self._train_steps % self.TARGET_UPDATE_FREQ == 0:
            self._update_target_network()

        self.base.add_experience((cur_buf, action, float(reward), next_buf))

    def mutate_lora(self, sigma: float = 0.02) -> None:
        with torch.no_grad():
            for p in (self.lora_A_in, self.lora_B_in,
                      self.lora_A_out, self.lora_B_out):
                p.data.add_(torch.randn_like(p.data) * sigma)
        self._update_target_network()

    def reset_lora_optimizer(self) -> None:
        self._lora_optimizer = optim.Adam(
            [self.lora_A_in, self.lora_B_in, self.lora_A_out, self.lora_B_out],
            lr=self._lr_lora,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_lora(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "lora_A_in":    self.lora_A_in.data,
                "lora_B_in":    self.lora_B_in.data,
                "lora_A_out":   self.lora_A_out.data,
                "lora_B_out":   self.lora_B_out.data,
                "target_A_in":  self._target_A_in,
                "target_B_in":  self._target_B_in,
                "target_A_out": self._target_A_out,
                "target_B_out": self._target_B_out,
                "h":            self._h,
                "train_steps":  self._train_steps,
            },
            path,
        )

    def load_lora(self, path: Path) -> None:
        if not path.exists():
            return
        sd = torch.load(path, map_location=_DEVICE, weights_only=True)
        for name in ("lora_A_in", "lora_B_in", "lora_A_out", "lora_B_out"):
            if name not in sd:
                continue
            saved = sd[name].to(_DEVICE)
            param = getattr(self, name)
            if tuple(param.shape) == tuple(saved.shape):
                param.data.copy_(saved)
        for tname, key in (
            ("_target_A_in",  "target_A_in"),
            ("_target_B_in",  "target_B_in"),
            ("_target_A_out", "target_A_out"),
            ("_target_B_out", "target_B_out"),
        ):
            if key in sd:
                setattr(self, tname, sd[key].to(_DEVICE).clone())
        if "h" in sd:
            self._h = sd["h"]
        if "train_steps" in sd:
            self._train_steps = int(sd["train_steps"])


class _CategoricalRoleController(CategoricalRNNController):
    """Mixin exposing ``save_table``/``load_table`` for API parity with the
    scalar roles (see :class:`~worldsim.ai.rnn.controller._RoleController`).
    """

    def __init__(
        self,
        role:       str,
        n_inputs:   int,
        n_outputs:  int,
        hidden_dim: int,
        *,
        table_path: Optional[str | Path] = None,
        epsilon:    float,
        gamma:      float,
        seed:       Optional[int] = None,
    ) -> None:
        super().__init__(
            role, n_inputs, n_outputs, hidden_dim,
            epsilon=epsilon, gamma=gamma, seed=seed,
        )
        self.table_path = Path(table_path) if table_path else None
        if self.table_path:
            self.load_table(self.table_path)

    def save_table(self, path: Optional[str | Path] = None) -> None:
        target = Path(path) if path else self.table_path
        if target is None:
            return
        self.save_lora(target.with_suffix(".lora.pt"))
        self.table_path = target

    def load_table(self, path: str | Path) -> None:
        self.load_lora(Path(path).with_suffix(".lora.pt"))
        self.table_path = Path(path)
