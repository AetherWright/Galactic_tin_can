"""Shared, stateless, batched network for ground-division control.

Ground divisions are numerous (dozens per nation, thousands of tiny forward
passes per fifth across a whole galaxy) and short-lived — templates get
destroyed in combat and rebuilt constantly, so per-division recurrent memory
(the GRU + rolling buffer every other role uses, see ``.unified``) doesn't
fit: there's no stable identity to carry hidden state across. Instead this
module is **stateless and batched**: one shared feedforward trunk scores
every division a nation has (or every division in the whole simulation) in a
single forward call, with the same per-nation LoRA specialisation used
everywhere else in the unified-trunk family.

This replaces the previous per-division controller pair (one tiny
``SimplePerceptron``/``NativePerceptron`` for posture, one for movement —
see ``worldsim.ai.native``) with **one** richer action head; see
``worldsim.military.divisions`` for the action list and feature vector, and
``worldsim.military.combat`` for how a nation's divisions are batched into
one call per fifth.

Architecture
------------
::

    x = in_proj(features) + coreLoRA(features)     # [B, HIDDEN]
    h = relu(hidden(relu(x)))                       # shared, not LoRA'd
    out = head(h) + headLoRA(h)                     # [B, N_ACTIONS]

``in_proj``/``hidden``/``head`` are shared by every nation; ``in_proj`` and
``head`` each get a small per-nation LoRA adapter (:class:`DivisionLoRABank`),
mirroring the core/head split in ``.unified`` — ``hidden`` stays
nation-invariant, same rationale as the unified trunk's un-LoRA'd GRU.

Thread safety mirrors the unified trunk's lock-free design (see
``.unified.UnifiedRoleView._forward``): every shared-trunk parameter is read
through ``.detach()`` during a nation's online batched training step, so
``loss.backward()`` only ever populates ``.grad`` on that nation's own LoRA
tensors. The trunk itself only trains from the pooled replay buffer via
:func:`step_division_trunk`, called once per century from the main thread —
concurrent nations never need to lock each other out.
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

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Shared trunk width — divisions' feature vector is far narrower than the
# unified trunk's (no galactic control grid), so a smaller hidden size keeps
# the per-call cost trivial even batched over hundreds of divisions.
DIVISION_HIDDEN_DIM: int = 64

_GAMMA: float = 0.95
_BATCH_SIZE: int = 128
_REPLAY_CAP: int = 8_000


class DivisionTrunkModel(nn.Module):
    """The single shared feedforward trunk, sized on first use."""

    def __init__(self, n_inputs: int, n_actions: int) -> None:
        super().__init__()
        self.n_inputs = n_inputs
        self.n_actions = n_actions
        self.in_proj = nn.Linear(n_inputs, DIVISION_HIDDEN_DIM)
        self.hidden = nn.Linear(DIVISION_HIDDEN_DIM, DIVISION_HIDDEN_DIM)
        self.head = nn.Linear(DIVISION_HIDDEN_DIM, n_actions)
        self.to(_DEVICE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Base forward (no LoRA) — used by :func:`step_division_trunk`."""
        h1 = F.relu(self.in_proj(x))
        h2 = F.relu(self.hidden(h1))
        return self.head(h2)


_TRUNK: Optional[DivisionTrunkModel] = None
_TRUNK_LOCK = threading.Lock()
_TRUNK_OPTIMIZER: Optional[optim.Optimizer] = None
_TRUNK_DIMS: Optional[Tuple[int, int]] = None
_BASE_LR: float = 1e-4


def get_division_trunk(n_inputs: int, n_actions: int) -> DivisionTrunkModel:
    """Return the process-wide singleton division trunk, sized on first use.

    Every nation's division bank shares the same ``(n_inputs, n_actions)``
    shape (see ``worldsim.military.divisions``), so unlike the unified
    trunk's per-role lazy registry, this trunk has exactly one shape for its
    whole lifetime — sized by whichever caller constructs it first.
    """
    global _TRUNK, _TRUNK_OPTIMIZER, _TRUNK_DIMS
    with _TRUNK_LOCK:
        if _TRUNK is None:
            _TRUNK = DivisionTrunkModel(n_inputs, n_actions)
            _TRUNK_OPTIMIZER = optim.Adam(_TRUNK.parameters(), lr=_BASE_LR)
            _TRUNK_DIMS = (n_inputs, n_actions)
        return _TRUNK


def reset_division_trunk() -> None:
    """Discard the singleton trunk (tests only — forces a clean rebuild)."""
    global _TRUNK, _TRUNK_OPTIMIZER, _TRUNK_DIMS
    with _TRUNK_LOCK:
        _TRUNK = None
        _TRUNK_OPTIMIZER = None
        _TRUNK_DIMS = None


# ---------------------------------------------------------------------------
# Shared replay buffer — trains the trunk from every nation's division batches
# ---------------------------------------------------------------------------

_ReplayItem = Tuple[List[float], int, float, List[float]]

_REPLAY: Deque[_ReplayItem] = deque(maxlen=_REPLAY_CAP)
_REPLAY_LOCK = threading.Lock()


def _add_experience(exp: _ReplayItem) -> None:
    with _REPLAY_LOCK:
        _REPLAY.append(exp)


def step_division_trunk(n_steps: int = 4) -> None:
    """Run *n_steps* SGD updates on the shared division trunk from the replay.

    Call this once per century from the main simulation thread, mirroring
    ``worldsim.ai.rnn.unified.step_trunk`` — never during threaded nation
    turns, so it never races the per-nation batched training step.
    """
    if _TRUNK is None:
        return
    trunk = _TRUNK
    assert _TRUNK_OPTIMIZER is not None
    with _REPLAY_LOCK:
        replay_snap = list(_REPLAY)

    batch_size = auto_batch_size(
        (2 * (trunk.n_inputs) + 4) * 4 * 4,
        device="vram" if _DEVICE.type == "cuda" else "ram",
        fraction=0.10,
        lo=32,
        hi=512,
        default=_BATCH_SIZE,
    )
    if len(replay_snap) < min(_BATCH_SIZE, batch_size):
        return

    for _ in range(n_steps):
        batch = random.sample(replay_snap, min(batch_size, len(replay_snap)))
        states, actions, rewards, next_states = zip(*batch)
        states_t = torch.tensor(states, dtype=torch.float32, device=_DEVICE)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=_DEVICE)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=_DEVICE)
        actions_t = torch.tensor(actions, dtype=torch.long, device=_DEVICE)

        with torch.no_grad():
            next_q = trunk(next_states_t)
            target = rewards_t + _GAMMA * next_q.max(dim=1).values
        cur_q = trunk(states_t)
        pred = cur_q.gather(1, actions_t.unsqueeze(1)).squeeze(1)
        loss = F.mse_loss(pred, target)

        _TRUNK_OPTIMIZER.zero_grad()
        loss.backward()
        _TRUNK_OPTIMIZER.step()


def save_division_trunk(path: Path) -> None:
    if _TRUNK is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_TRUNK.state_dict(), path)


def load_division_trunk(path: Path, n_inputs: int, n_actions: int) -> None:
    path = Path(path)
    if not path.exists():
        return
    sd = torch.load(path, map_location=_DEVICE, weights_only=True)
    get_division_trunk(n_inputs, n_actions).load_state_dict(sd, strict=False)


# ---------------------------------------------------------------------------
# Per-nation LoRA bank — batched inference + training for every division
# a nation owns in one call
# ---------------------------------------------------------------------------

class DivisionLoRABank:
    """One nation's division-controller LoRA, batched across all its divisions.

    Unlike :class:`~worldsim.ai.rnn.unified.UnifiedLoRABank` (one core LoRA +
    one head LoRA *per role*), there is only one role here — every division a
    nation owns is scored by the same adapted trunk in the same batched call,
    so a single core+head LoRA pair per nation is all this needs.
    """

    _LORA_R: int = 8
    _LORA_ALPHA: float = 16.0
    _LR_LORA: float = 3e-4
    TARGET_UPDATE_FREQ: int = 16

    def __init__(
        self,
        n_inputs: int,
        n_actions: int,
        *,
        epsilon: float = 0.15,
        gamma: float = _GAMMA,
        lora_r: int = _LORA_R,
        lora_alpha: float = _LORA_ALPHA,
        lr_lora: float = _LR_LORA,
        seed: Optional[int] = None,
    ) -> None:
        self.n_inputs = n_inputs
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.gamma = gamma
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self._rng = random.Random(seed)

        self.trunk = get_division_trunk(n_inputs, n_actions)

        self.lora_A_in = nn.Parameter(torch.empty(lora_r, n_inputs, device=_DEVICE))
        self.lora_B_in = nn.Parameter(torch.zeros(DIVISION_HIDDEN_DIM, lora_r, device=_DEVICE))
        nn.init.kaiming_uniform_(self.lora_A_in, a=math.sqrt(5))

        self.lora_A_out = nn.Parameter(torch.empty(lora_r, DIVISION_HIDDEN_DIM, device=_DEVICE))
        self.lora_B_out = nn.Parameter(torch.zeros(n_actions, lora_r, device=_DEVICE))
        nn.init.kaiming_uniform_(self.lora_A_out, a=math.sqrt(5))

        self._lr_lora = lr_lora
        self._optimizer = optim.Adam(
            [self.lora_A_in, self.lora_B_in, self.lora_A_out, self.lora_B_out],
            lr=lr_lora,
        )

        with torch.no_grad():
            self._target_A_in = self.lora_A_in.data.clone()
            self._target_B_in = self.lora_B_in.data.clone()
            self._target_A_out = self.lora_A_out.data.clone()
            self._target_B_out = self.lora_B_out.data.clone()
        self._train_steps: int = 0

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """Online batched forward: trunk read through ``.detach()`` so this
        nation's backward pass only ever touches its own LoRA tensors."""
        scale = self.lora_alpha / self.lora_r
        trunk = self.trunk

        proj_in = F.linear(x, trunk.in_proj.weight.detach(), trunk.in_proj.bias.detach())
        lora_in = (x @ self.lora_A_in.T) @ self.lora_B_in.T * scale
        h1 = F.relu(proj_in + lora_in)
        h2 = F.relu(F.linear(h1, trunk.hidden.weight.detach(), trunk.hidden.bias.detach()))

        proj_out = F.linear(h2, trunk.head.weight.detach(), trunk.head.bias.detach())
        lora_out = (h2 @ self.lora_A_out.T) @ self.lora_B_out.T * scale
        return proj_out + lora_out

    def _forward_target(self, x: torch.Tensor) -> torch.Tensor:
        """Forward using the frozen TARGET LoRA (Double DQN bootstrap)."""
        scale = self.lora_alpha / self.lora_r
        trunk = self.trunk

        proj_in = F.linear(x, trunk.in_proj.weight.detach(), trunk.in_proj.bias.detach())
        lora_in = (x @ self._target_A_in.T) @ self._target_B_in.T * scale
        h1 = F.relu(proj_in + lora_in)
        h2 = F.relu(F.linear(h1, trunk.hidden.weight.detach(), trunk.hidden.bias.detach()))

        proj_out = F.linear(h2, trunk.head.weight.detach(), trunk.head.bias.detach())
        lora_out = (h2 @ self._target_A_out.T) @ self._target_B_out.T * scale
        return proj_out + lora_out

    def _update_target_network(self) -> None:
        with torch.no_grad():
            self._target_A_in.copy_(self.lora_A_in.data)
            self._target_B_in.copy_(self.lora_B_in.data)
            self._target_A_out.copy_(self.lora_A_out.data)
            self._target_B_out.copy_(self.lora_B_out.data)

    # ------------------------------------------------------------------
    # Public batched API
    # ------------------------------------------------------------------

    def choose_actions(
        self,
        states: Sequence[Sequence[float]],
        *,
        valid_masks: Optional[Sequence[Optional[Sequence[bool]]]] = None,
    ) -> List[int]:
        """Return one action index per row of *states* — ONE forward call
        scores every division passed in, however many there are."""
        if not states:
            return []
        x = torch.tensor(list(states), dtype=torch.float32, device=_DEVICE)
        with torch.no_grad():
            scores = self._forward(x).tolist()
        actions: List[int] = []
        for i, row in enumerate(scores):
            mask = valid_masks[i] if valid_masks else None
            if self._rng.random() < self.epsilon:
                if mask:
                    valid = [j for j in range(len(row)) if mask[j]]
                    actions.append(self._rng.choice(valid) if valid else 0)
                else:
                    actions.append(self._rng.randrange(len(row)))
                continue
            best_idx, best_val = 0, float("-inf")
            for j, val in enumerate(row):
                if mask and not mask[j]:
                    continue
                if val > best_val:
                    best_idx, best_val = j, val
            actions.append(best_idx)
        return actions

    def train_batch(self, transitions: Sequence[_ReplayItem]) -> None:
        """One batched Double-DQN update over every ``(state, action, reward,
        next_state)`` transition passed in — a single backward pass trains
        every division's transition this fifth, instead of the old one
        perceptron-update-per-division loop."""
        if not transitions:
            return
        states, actions, rewards, next_states = zip(*transitions)
        states_t = torch.tensor(states, dtype=torch.float32, device=_DEVICE)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=_DEVICE)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=_DEVICE)
        actions_t = torch.tensor(
            [max(0, min(a, self.n_actions - 1)) for a in actions],
            dtype=torch.long, device=_DEVICE,
        )

        with torch.no_grad():
            next_q_online = self._forward(next_states_t)
            best_next = next_q_online.argmax(dim=1)
            next_q_target = self._forward_target(next_states_t)
            next_val = next_q_target.gather(1, best_next.unsqueeze(1)).squeeze(1)
            target = rewards_t + self.gamma * next_val

        cur_q = self._forward(states_t)
        pred = cur_q.gather(1, actions_t.unsqueeze(1)).squeeze(1)
        loss = F.mse_loss(pred, target)

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        self._train_steps += 1
        if self._train_steps % self.TARGET_UPDATE_FREQ == 0:
            self._update_target_network()

        with _REPLAY_LOCK:
            for state, action, reward, next_state in transitions:
                _REPLAY.append((list(state), int(action), float(reward), list(next_state)))

    def mutate_lora(self, sigma: float = 0.02) -> None:
        with torch.no_grad():
            for p in (self.lora_A_in, self.lora_B_in, self.lora_A_out, self.lora_B_out):
                p.data.add_(torch.randn_like(p.data) * sigma)
        self._update_target_network()

    def reset_optimizer(self) -> None:
        self._optimizer = optim.Adam(
            [self.lora_A_in, self.lora_B_in, self.lora_A_out, self.lora_B_out],
            lr=self._lr_lora,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_lora(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "lora_A_in": self.lora_A_in.data,
                "lora_B_in": self.lora_B_in.data,
                "lora_A_out": self.lora_A_out.data,
                "lora_B_out": self.lora_B_out.data,
                "target_A_in": self._target_A_in,
                "target_B_in": self._target_B_in,
                "target_A_out": self._target_A_out,
                "target_B_out": self._target_B_out,
                "train_steps": self._train_steps,
            },
            path,
        )

    def load_lora(self, path: Path) -> None:
        path = Path(path)
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
        for name in ("target_A_in", "target_B_in", "target_A_out", "target_B_out"):
            attr = f"_{name}"
            if name in sd and sd[name].shape == getattr(self, attr).shape:
                setattr(self, attr, sd[name].to(_DEVICE).clone())
        if "train_steps" in sd:
            self._train_steps = int(sd["train_steps"])

    def save_table(self, path: Path) -> None:
        self.save_lora(Path(path).with_suffix(".lora.pt"))

    def load_table(self, path: Path) -> None:
        self.load_lora(Path(path).with_suffix(".lora.pt"))
