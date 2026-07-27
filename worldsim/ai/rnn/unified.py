"""One shared trunk for every AI role, with per-nation LoRA specialisation.

Phase 1 (see the package docstring) gave every role its own GRU trunk,
keyed by ``role × input_dim × output_dim`` (``base.py``'s ``RoleBaseModel`` /
``get_role_model`` registry). This module replaces that with a **single**
shared trunk plus one output head per role, so every role's experience
trains the same recurrent weights instead of 13+ independent ones.

Input shape
-----------
Every role's forward pass is fed one flat vector per timestep, built as::

    combined = core(16) ++ extra(role-specific)

``core`` is :meth:`~worldsim.nations.nation.Nation.unified_core_state` — the
same 8 fields as ``reward_snapshot()`` (economy/stability/military/tech/
infrastructure/population/star_count/fleet_count), NN-scaled, plus the 8
:class:`~worldsim.society.culture.Culture` traits. It's identical for every
role, which is what lets a nation's culture influence every decision the
nation makes. ``extra`` is whatever feature vector that role already built
before this change (untouched — see each role wrapper in ``roles.py``).

Architecture
------------
::

    x = core_proj(core) + extra_proj[role](extra) + role_embedding[role]
    gru_out, h' = gru(x_seq, h)              # ONE shared GRU
    out = head[role](gru_out[:, -1])         # per-role output layer

``core_proj``, ``gru`` and ``role_embedding`` are shared by every role and
every nation. ``extra_proj[role]`` and ``head[role]`` are created lazily on
first use (mirrors the old ``get_role_model`` lazy-registry pattern) and are
also shared across nations — they encode role-specific situational
features/action-space shape, not per-nation personality.

Per-nation LoRA
----------------
Specialisation is split across the two places behaviour actually diverges
per nation:

* **Core LoRA** (one per nation): adapts ``core_proj``. Since ``core``
  includes culture, this is where a nation learns how much its culture
  should shape its behaviour.
* **Head LoRA** (one per nation × role): adapts that role's ``head``,
  exactly like phase 1's per-role output LoRA.

``extra_proj``/``role_embedding`` are not LoRA'd — nation-invariant.

Training mirrors phase 1 exactly (Double DQN with a hard-copied target
network for scalar heads; C51 categorical cross-entropy against a projected
Bellman target for the ``events`` head), just operating on the split
core/extra input and the narrower LoRA surface described above.
"""
from __future__ import annotations

import math
import random
import threading
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ...core.memory import auto_batch_size

# Prefer CUDA when available; otherwise fall back to CPU.
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Number of past input states kept in the rolling buffer.
BUFFER_SIZE: int = 5

# Canonical shared "core" input — see Nation.unified_core_state().
CORE_DIM: int = 16

# Shared trunk width. Sized to the largest role that existed pre-unification
# (war's GRU was 128-wide).
HIDDEN_DIM: int = 128

# C51 distributional support for the "events" categorical head.
N_ATOMS: int = 51
V_MIN: float = -50.0
V_MAX: float = 50.0
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
    atoms so the result stays a valid distribution over the same support.
    """
    batch = next_dist.shape[0]
    n_atoms = support.shape[0]
    delta_z = (V_MAX - V_MIN) / (n_atoms - 1)

    tz = rewards.view(batch, 1) + gamma * support.view(1, n_atoms)
    tz = tz.clamp(V_MIN, V_MAX)
    b = (tz - V_MIN) / delta_z
    lower = b.floor()
    upper = b.ceil()
    same = lower == upper

    upper_weight = torch.where(same, torch.zeros_like(b), upper - b)
    lower_weight = torch.where(same, torch.ones_like(b), b - lower)

    lower_idx = lower.long().clamp(0, n_atoms - 1)
    upper_idx = upper.long().clamp(0, n_atoms - 1)

    projected = torch.zeros(batch, n_atoms, device=next_dist.device)
    projected.scatter_add_(1, lower_idx, next_dist * upper_weight)
    projected.scatter_add_(1, upper_idx, next_dist * lower_weight)
    return projected


# ---------------------------------------------------------------------------
# Shared trunk — one instance for the whole process
# ---------------------------------------------------------------------------

class UnifiedTrunkModel(nn.Module):
    """The single shared GRU trunk, with per-role adapters/heads created lazily."""

    def __init__(self) -> None:
        super().__init__()
        self.core_proj = nn.Linear(CORE_DIM, HIDDEN_DIM)
        self.gru = nn.GRU(HIDDEN_DIM, HIDDEN_DIM, batch_first=True)
        self.role_embedding = nn.ParameterDict()
        self.extra_proj = nn.ModuleDict()
        self.heads = nn.ModuleDict()
        self._extra_dim: Dict[str, int] = {}
        self._n_outputs: Dict[str, int] = {}
        self._categorical: Dict[str, bool] = {}
        self.to(_DEVICE)

    def ensure_role(
        self, role: str, extra_dim: int, n_outputs: int, *, categorical: bool = False
    ) -> None:
        """Lazily create (or resize) the adapter/head for *role*.

        Rebuilding a role's ``extra_proj`` (e.g. when ``war``'s live nation
        count changes) only touches that role — every other role, and the
        shared ``core_proj``/``gru``/``role_embedding``, are untouched.

        Any parameter created here is registered with the trunk optimizer
        (:func:`_register_new_trunk_params`) — the optimizer is built once,
        early, over whatever parameters exist *then*; every role added
        afterwards needs its own ``add_param_group`` call or ``step_trunk``
        would silently never update it.
        """
        extra_dim = max(1, extra_dim)
        new_params: List[nn.Parameter] = []
        if role not in self.extra_proj or self._extra_dim.get(role) != extra_dim:
            was_resize = role in self._extra_dim and self._extra_dim[role] != extra_dim
            layer = nn.Linear(extra_dim, HIDDEN_DIM).to(_DEVICE)
            self.extra_proj[role] = layer
            self._extra_dim[role] = extra_dim
            new_params.extend(layer.parameters())
            if was_resize:
                # Every buffered replay entry for this role was recorded at
                # the *previous* extra width (e.g. TorchWarAI.set_allies_dimension
                # changing "war"'s width as nations are added/removed) — now
                # incompatible with the just-updated self._extra_dim[role],
                # which would crash step_trunk()'s reshape on the next call.
                _purge_replay_for_role(role)
        if role not in self.role_embedding:
            emb = nn.Parameter(torch.zeros(HIDDEN_DIM, device=_DEVICE))
            self.role_embedding[role] = emb
            new_params.append(emb)
        if role not in self.heads:
            out_dim = n_outputs * N_ATOMS if categorical else n_outputs
            layer = nn.Linear(HIDDEN_DIM, out_dim).to(_DEVICE)
            self.heads[role] = layer
            self._n_outputs[role] = n_outputs
            self._categorical[role] = categorical
            new_params.extend(layer.parameters())
        if new_params:
            _register_new_trunk_params(new_params)

    def extra_dim(self, role: str) -> int:
        return self._extra_dim[role]

    def forward(
        self,
        role: str,
        core_seq: torch.Tensor,    # [1, T, CORE_DIM]
        extra_seq: torch.Tensor,   # [1, T, extra_dim]
        h: torch.Tensor,           # [1, 1, HIDDEN_DIM]
    ) -> torch.Tensor:
        """Base forward (no LoRA) — used by :meth:`update_base`."""
        T = core_seq.shape[1]
        core_flat = core_seq.reshape(T, CORE_DIM)
        extra_flat = extra_seq.reshape(T, self._extra_dim[role])
        x = (
            self.core_proj(core_flat)
            + self.extra_proj[role](extra_flat)
            + self.role_embedding[role]
        ).unsqueeze(0)
        gru_out, _ = self.gru(x, h)
        out = self.heads[role](gru_out[:, -1])
        if self._categorical[role]:
            out = out.view(-1, self._n_outputs[role], N_ATOMS)
            out = F.log_softmax(out, dim=-1)
        return out


_TRUNK: Optional[UnifiedTrunkModel] = None
_TRUNK_LOCK = threading.Lock()
_TRUNK_OPTIMIZER: Optional[optim.Optimizer] = None
_BASE_LR: float = 1e-4


def get_trunk() -> UnifiedTrunkModel:
    """Return the process-wide singleton trunk, creating it on first use."""
    global _TRUNK, _TRUNK_OPTIMIZER
    with _TRUNK_LOCK:
        if _TRUNK is None:
            _TRUNK = UnifiedTrunkModel()
            _TRUNK_OPTIMIZER = optim.Adam(_TRUNK.parameters(), lr=_BASE_LR)
        return _TRUNK


def _trunk_optimizer() -> optim.Optimizer:
    get_trunk()
    assert _TRUNK_OPTIMIZER is not None
    return _TRUNK_OPTIMIZER


def _register_new_trunk_params(params: List[nn.Parameter]) -> None:
    """Add freshly created trunk parameters to the trunk optimizer.

    ``optim.Adam(trunk.parameters(), ...)`` only ever sees the parameters
    that existed at the moment it was constructed. Every role's
    ``extra_proj``/``head``/``role_embedding`` is created lazily, after that
    optimizer already exists, so each batch of new parameters needs its own
    ``add_param_group`` call or they'd never receive a gradient step from
    :func:`step_trunk`.
    """
    if _TRUNK_OPTIMIZER is None:
        # Trunk (and its optimizer) haven't been constructed yet — get_trunk()
        # will pick up these parameters when it builds the optimizer.
        return
    _TRUNK_OPTIMIZER.add_param_group({"params": list(params)})


def reset_trunk() -> None:
    """Discard the singleton trunk (tests only — forces a clean rebuild)."""
    global _TRUNK, _TRUNK_OPTIMIZER
    with _TRUNK_LOCK:
        _TRUNK = None
        _TRUNK_OPTIMIZER = None


# ---------------------------------------------------------------------------
# Shared replay buffer — trains the trunk from every role's experience
# ---------------------------------------------------------------------------

_ReplayItem = Tuple[str, List[List[float]], int, float, List[List[float]]]

_REPLAY_CAP: int = 16_000
_REPLAY: Deque[_ReplayItem] = deque(maxlen=_REPLAY_CAP)
_REPLAY_LOCK = threading.Lock()
_BATCH_SIZE: int = 128
_GAMMA: float = 0.95

def add_experience(exp: _ReplayItem) -> None:
    with _REPLAY_LOCK:
        _REPLAY.append(exp)


def clear_replay() -> None:
    """Drop every buffered experience (up to :data:`_REPLAY_CAP` = 16 000
    entries). Call before persisting a run's models — this is training
    scratch space, not learned state, and there's no reason to keep it (or
    the RAM it holds) alive across a checkpoint."""
    with _REPLAY_LOCK:
        _REPLAY.clear()


def _purge_replay_for_role(role: str) -> None:
    """Drop every buffered replay entry for *role*.

    Called from :meth:`UnifiedTrunkModel.ensure_role` when a role's shared
    ``extra_proj`` is resized — every entry buffered before the resize was
    recorded at the previous (now incompatible) extra width, and
    :func:`step_trunk` reshaping them against the new width would crash.
    """
    with _REPLAY_LOCK:
        kept = [item for item in _REPLAY if item[0] != role]
        _REPLAY.clear()
        _REPLAY.extend(kept)


def step_trunk(n_steps: int = 4) -> None:
    """Run *n_steps* SGD updates on the shared trunk from the replay buffer.

    Call this from the main simulation thread (not during threaded nation
    turns) once per century — mirrors phase 1's ``step_all_base_models``.
    """
    trunk = get_trunk()
    with _REPLAY_LOCK:
        replay_snap = list(_REPLAY)

    batch_size = auto_batch_size(
        (2 * BUFFER_SIZE * (CORE_DIM + 32) + 12 * HIDDEN_DIM) * 4 * 4,
        device="vram" if _DEVICE.type == "cuda" else "ram",
        fraction=0.10,
        lo=32,
        hi=512,
        default=_BATCH_SIZE,
    )
    if len(replay_snap) < min(_BATCH_SIZE, batch_size):
        return

    h0 = torch.zeros(1, 1, HIDDEN_DIM, device=_DEVICE)
    trunk_optimizer = _trunk_optimizer()

    for _ in range(n_steps):
        batch = random.sample(replay_snap, min(batch_size, len(replay_snap)))
        total_loss = torch.zeros(1, device=_DEVICE)
        for role, cur_buf, action, reward, next_buf in batch:
            if role not in trunk.heads:
                continue
            extra_dim = trunk.extra_dim(role)
            cur_core, cur_extra = _split(cur_buf, extra_dim)
            next_core, next_extra = _split(next_buf, extra_dim)
            cur_core_t = torch.tensor(cur_core, dtype=torch.float32, device=_DEVICE).unsqueeze(0)
            cur_extra_t = torch.tensor(cur_extra, dtype=torch.float32, device=_DEVICE).unsqueeze(0)
            next_core_t = torch.tensor(next_core, dtype=torch.float32, device=_DEVICE).unsqueeze(0)
            next_extra_t = torch.tensor(next_extra, dtype=torch.float32, device=_DEVICE).unsqueeze(0)

            if trunk._categorical[role]:
                with torch.no_grad():
                    next_probs = trunk.forward(role, next_core_t, next_extra_t, h0).exp()
                    expected = (next_probs * _SUPPORT).sum(dim=-1)
                    best_next = int(expected.argmax(dim=1).item())
                    next_dist = next_probs[:, best_next]
                    reward_t = torch.tensor([reward], dtype=torch.float32, device=_DEVICE)
                    target_dist = project_categorical(next_dist, reward_t, _GAMMA)
                cur_log_probs = trunk.forward(role, cur_core_t, cur_extra_t, h0)
                if 0 <= action < cur_log_probs.shape[1]:
                    log_p_a = cur_log_probs[0, action]
                    total_loss = total_loss - (target_dist[0] * log_p_a).sum()
            else:
                with torch.no_grad():
                    next_q = trunk.forward(role, next_core_t, next_extra_t, h0)
                    tv = reward + _GAMMA * float(next_q.max())
                cur_q = trunk.forward(role, cur_core_t, cur_extra_t, h0)
                target = cur_q.detach().clone()
                if 0 <= action < target.shape[1]:
                    target[0, action] = tv
                total_loss = total_loss + F.mse_loss(cur_q, target)

        total_loss = total_loss / len(batch)
        trunk_optimizer.zero_grad()
        total_loss.backward()
        trunk_optimizer.step()


def _split(buf: List[List[float]], extra_dim: int) -> Tuple[List[List[float]], List[List[float]]]:
    core = [row[:CORE_DIM] for row in buf]
    extra = [row[CORE_DIM:CORE_DIM + extra_dim] for row in buf]
    return core, extra


def save_trunk(path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(get_trunk().state_dict(), path)


def load_trunk(path: Path) -> None:
    path = Path(path)
    if not path.exists():
        return
    sd = torch.load(path, map_location=_DEVICE, weights_only=True)
    get_trunk().load_state_dict(sd, strict=False)


# ---------------------------------------------------------------------------
# Per-nation, per-role view — the public controller-like API
# ---------------------------------------------------------------------------

class UnifiedRoleView:
    """Nation-specific, role-specific controller bound to the shared trunk.

    Exposes the exact same surface phase 1's per-role controllers did
    (``predict`` / ``predict_prob`` / ``choose_action`` / ``train`` /
    ``save_table`` / ``load_table`` / ``mutate_lora`` /
    ``reset_lora_optimizer``), so nothing at any call site needs to change —
    only how these objects get constructed (see ``roles.py``).
    """

    _LORA_R: int = 8
    _LORA_ALPHA: float = 16.0
    _GAMMA: float = 0.95
    _EPSILON: float = 0.10
    _LR_LORA: float = 3e-4
    TARGET_UPDATE_FREQ: int = 16

    def __init__(
        self,
        bank: "UnifiedLoRABank",
        role: str,
        n_inputs: int,
        n_outputs: int,
        *,
        categorical: bool = False,
        epsilon: float = _EPSILON,
        gamma: float = _GAMMA,
        lora_r: int = _LORA_R,
        lora_alpha: float = _LORA_ALPHA,
        lr_lora: float = _LR_LORA,
        seed: Optional[int] = None,
    ) -> None:
        self.bank = bank
        self.role = role
        self.n_inputs = n_inputs     # role's "extra" width (unchanged external name)
        self.n_outputs = n_outputs
        self.categorical = categorical
        self.epsilon = epsilon
        self.gamma = gamma
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self._rng = random.Random(seed)

        self.trunk = get_trunk()
        self.trunk.ensure_role(role, n_inputs, n_outputs, categorical=categorical)
        bank.roles[role] = self

        out_dim = n_outputs * N_ATOMS if categorical else n_outputs
        self.lora_A_head = nn.Parameter(torch.empty(lora_r, HIDDEN_DIM, device=_DEVICE))
        self.lora_B_head = nn.Parameter(torch.zeros(out_dim, lora_r, device=_DEVICE))
        nn.init.kaiming_uniform_(self.lora_A_head, a=math.sqrt(5))

        self._lr_lora = lr_lora
        self._head_optimizer = optim.Adam([self.lora_A_head, self.lora_B_head], lr=lr_lora)

        with torch.no_grad():
            self._target_A_head = self.lora_A_head.data.clone()
            self._target_B_head = self.lora_B_head.data.clone()
        self._train_steps: int = 0

        self._h: torch.Tensor = torch.zeros(1, 1, HIDDEN_DIM, device=_DEVICE)
        combined_dim = CORE_DIM + n_inputs
        self._buffer: Deque[List[float]] = deque(
            [[0.0] * combined_dim] * BUFFER_SIZE, maxlen=BUFFER_SIZE
        )

    def reset_runtime_state(self) -> None:
        """Zero this role's transient GRU hidden state and rolling buffer.

        Neither is *learned* state — the hidden state is just recent
        context carried between calls, and the buffer is the last
        :data:`BUFFER_SIZE` observed inputs. Call this before persisting a
        nation (see ``worldsim.ai.persistence``) so a checkpoint doesn't
        needlessly bloat with per-nation runtime tensors, and so a nation
        loaded from that checkpoint starts fresh rather than resuming
        mid-context from wherever the previous run happened to stop.
        """
        self._h = torch.zeros(1, 1, HIDDEN_DIM, device=_DEVICE)
        combined_dim = CORE_DIM + self.n_inputs
        self._buffer = deque(
            [[0.0] * combined_dim] * BUFFER_SIZE, maxlen=BUFFER_SIZE
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pad_extra(self, extra: Sequence[float]) -> List[float]:
        vec = list(extra)[: self.n_inputs]
        vec += [0.0] * max(0, self.n_inputs - len(vec))
        return vec

    def _combined(self, extra: Sequence[float]) -> List[float]:
        return list(self.bank.core_state()) + self._pad_extra(extra)

    def _split_seq(self, seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return seq[:, :, :CORE_DIM], seq[:, :, CORE_DIM:]

    @staticmethod
    def _buf_to_tensor(buf: Deque) -> torch.Tensor:
        return torch.tensor(list(buf), dtype=torch.float32, device=_DEVICE).unsqueeze(0)

    def _forward(self, seq: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Online forward: core_proj+coreLoRA, extra_proj, role_embedding, gru, head+headLoRA.

        Every shared-trunk parameter is read through ``.detach()`` here — the
        trunk (``core_proj``/``gru``/``extra_proj``/``role_embedding``/
        ``heads``) is only ever trained separately, from the pooled replay
        buffer, by :func:`step_trunk` on the main thread. Detaching means
        ``loss.backward()`` in :meth:`train` only ever populates ``.grad`` on
        this nation's own LoRA tensors, so concurrent nations' threads no
        longer race on shared trunk gradients and don't need a lock around
        their online training step.
        """
        core_seq, extra_seq = self._split_seq(seq)
        T = core_seq.shape[1]
        core_flat = core_seq.reshape(T, CORE_DIM)
        extra_flat = extra_seq.reshape(T, self.n_inputs)
        scale = self.lora_alpha / self.lora_r

        extra_layer = self.trunk.extra_proj[self.role]
        head = self.trunk.heads[self.role]

        proj_core = F.linear(
            core_flat, self.trunk.core_proj.weight.detach(), self.trunk.core_proj.bias.detach()
        )
        lora_core = (core_flat @ self.bank.lora_A_core.T) @ self.bank.lora_B_core.T * scale
        x = (
            proj_core + lora_core
            + F.linear(extra_flat, extra_layer.weight.detach(), extra_layer.bias.detach())
            + self.trunk.role_embedding[self.role].detach()
        ).unsqueeze(0)

        gru_params = {k: v.detach() for k, v in self.trunk.gru.named_parameters()}
        gru_out, h_new = torch.func.functional_call(self.trunk.gru, gru_params, (x, h))
        last = gru_out[:, -1]

        proj_head = F.linear(last, head.weight.detach(), head.bias.detach())
        lora_head = (last @ self.lora_A_head.T) @ self.lora_B_head.T * scale
        out = proj_head + lora_head
        if self.categorical:
            out = out.view(-1, self.n_outputs, N_ATOMS)
            out = F.log_softmax(out, dim=-1)
        return out, h_new

    def _forward_target(self, seq: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Forward using the frozen TARGET core+head LoRA (Double DQN / C51 bootstrap)."""
        core_seq, extra_seq = self._split_seq(seq)
        T = core_seq.shape[1]
        core_flat = core_seq.reshape(T, CORE_DIM)
        extra_flat = extra_seq.reshape(T, self.n_inputs)
        scale = self.lora_alpha / self.lora_r

        proj_core = F.linear(core_flat, self.trunk.core_proj.weight, self.trunk.core_proj.bias)
        lora_core = (core_flat @ self.bank._target_A_core.T) @ self.bank._target_B_core.T * scale
        x = (
            proj_core + lora_core
            + self.trunk.extra_proj[self.role](extra_flat)
            + self.trunk.role_embedding[self.role]
        ).unsqueeze(0)

        gru_out, _ = self.trunk.gru(x, h)
        last = gru_out[:, -1]

        head = self.trunk.heads[self.role]
        proj_head = F.linear(last, head.weight, head.bias)
        lora_head = (last @ self._target_A_head.T) @ self._target_B_head.T * scale
        out = proj_head + lora_head
        if self.categorical:
            out = out.view(-1, self.n_outputs, N_ATOMS)
            out = F.log_softmax(out, dim=-1)
        return out

    def _update_target_network(self) -> None:
        """Hard-copy the online head LoRA *and* the nation's online core LoRA
        into their respective frozen targets — the core is shared across
        every role for this nation but is trained continuously (every
        role's ``train()`` steps its Adam optimizer), so it needs the same
        periodic target sync as the head or ``_forward_target`` would keep
        bootstrapping off an increasingly stale core.
        """
        with torch.no_grad():
            self._target_A_head.copy_(self.lora_A_head.data)
            self._target_B_head.copy_(self.lora_B_head.data)
        self.bank.sync_core_target()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, extra: Sequence[float]) -> List[float]:
        """Forward pass; returns one raw score (or expected value) per action."""
        combined = self._combined(extra)
        self._buffer.append(combined)
        seq = self._buf_to_tensor(self._buffer)
        with torch.no_grad():
            out, h_new = self._forward(seq, self._h)
            if self.categorical:
                out = (out.exp() * _SUPPORT).sum(dim=-1)
        self._h = h_new.detach()
        return out[0].tolist()

    def predict_prob(self, extra: Sequence[float]) -> List[float]:
        raw = self.predict(extra)
        t = torch.tensor(raw)
        return torch.softmax(t, dim=0).tolist()

    def choose_action(
        self, extra: Sequence[float], valid_mask: Optional[List[bool]] = None
    ) -> int:
        if self.n_outputs <= 0:
            return 0
        scores = self.predict(extra)
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
        extra: Sequence[float],
        action: int,
        reward: float,
        next_extra: Sequence[float],
    ) -> None:
        if not (0 <= action < self.n_outputs):
            return

        cur_buf = list(self._buffer)
        next_combined = self._combined(next_extra)
        next_buf_deq: Deque[List[float]] = deque(self._buffer, maxlen=BUFFER_SIZE)
        next_buf_deq.append(next_combined)
        next_buf = list(next_buf_deq)

        cur_seq = torch.tensor(cur_buf, dtype=torch.float32, device=_DEVICE).unsqueeze(0)
        next_seq = torch.tensor(next_buf, dtype=torch.float32, device=_DEVICE).unsqueeze(0)
        reward_val = float(reward)
        h_ref = self._h

        if self.categorical:
            with torch.no_grad():
                next_online, _ = self._forward(next_seq, h_ref)
                next_probs_online = next_online.exp()
                expected_online = (next_probs_online * _SUPPORT).sum(dim=-1)
                best_next_act = int(expected_online.argmax(dim=1).item())

                next_target = self._forward_target(next_seq, h_ref)
                next_dist_target = next_target.exp()[:, best_next_act]
                reward_t = torch.tensor([reward_val], dtype=torch.float32, device=_DEVICE)
                target_dist = project_categorical(next_dist_target, reward_t, self.gamma)

            cur_out, _ = self._forward(cur_seq, h_ref)
            log_p_a = cur_out[0, action]
            loss = -(target_dist[0] * log_p_a).sum()
        else:
            with torch.no_grad():
                next_q_online, _ = self._forward(next_seq, h_ref)
                best_next_act = int(next_q_online.argmax(dim=1).item())
                next_q_target = self._forward_target(next_seq, h_ref)
                target_val = reward_val + self.gamma * float(next_q_target[0, best_next_act])

            cur_q, _ = self._forward(cur_seq, h_ref)
            target = cur_q.detach().clone()
            target[0, action] = target_val
            loss = F.mse_loss(cur_q, target)

        self.bank._core_optimizer.zero_grad()
        self._head_optimizer.zero_grad()
        loss.backward()
        self.bank._core_optimizer.step()
        self._head_optimizer.step()

        self._train_steps += 1
        if self._train_steps % self.TARGET_UPDATE_FREQ == 0:
            self._update_target_network()

        add_experience((self.role, cur_buf, action, reward_val, next_buf))

    def mutate_lora(self, sigma: float = 0.02) -> None:
        with torch.no_grad():
            for p in (self.lora_A_head, self.lora_B_head):
                p.data.add_(torch.randn_like(p.data) * sigma)
        self._update_target_network()

    def reset_lora_optimizer(self) -> None:
        self._head_optimizer = optim.Adam(
            [self.lora_A_head, self.lora_B_head], lr=self._lr_lora
        )

    # ------------------------------------------------------------------
    # Persistence — one file per (nation, role), same convention as phase 1
    # ------------------------------------------------------------------

    def save_lora(self, path: Path) -> None:
        """Persist the *learned* LoRA weights only.

        Deliberately excludes the GRU hidden state and rolling input buffer
        (see :meth:`reset_runtime_state`) — those are transient runtime
        context, not learned parameters, so a loaded nation always starts
        from a clean hidden state rather than resuming mid-context from
        wherever the saving run happened to stop.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "lora_A_head": self.lora_A_head.data,
                "lora_B_head": self.lora_B_head.data,
                "target_A_head": self._target_A_head,
                "target_B_head": self._target_B_head,
                "train_steps": self._train_steps,
            },
            path,
        )

    def load_lora(self, path: Path) -> None:
        path = Path(path)
        if not path.exists():
            return
        sd = torch.load(path, map_location=_DEVICE, weights_only=True)
        for name, tname in (("lora_A_head", "lora_A_head"), ("lora_B_head", "lora_B_head")):
            if tname not in sd:
                continue
            saved = sd[tname].to(_DEVICE)
            param = getattr(self, name)
            if tuple(param.shape) == tuple(saved.shape):
                param.data.copy_(saved)
        for tname, key in (
            ("_target_A_head", "target_A_head"),
            ("_target_B_head", "target_B_head"),
        ):
            if key in sd and sd[key].shape == getattr(self, tname).shape:
                setattr(self, tname, sd[key].to(_DEVICE).clone())
        if "train_steps" in sd:
            self._train_steps = int(sd["train_steps"])


class _TableRoleView(UnifiedRoleView):
    """Adds ``save_table``/``load_table`` compatibility (phase 1 naming)."""

    def __init__(self, *args, table_path: Optional[str | Path] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
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


# ---------------------------------------------------------------------------
# Per-nation LoRA bank — the shared "core" adapter + a role-view registry
# ---------------------------------------------------------------------------

class UnifiedLoRABank:
    """One nation's core LoRA plus every :class:`UnifiedRoleView` it owns.

    ``core_state_fn`` is a zero-arg callable returning the owning nation's
    current :meth:`~worldsim.nations.nation.Nation.unified_core_state` — kept
    as a callback (rather than a ``Nation`` import) to avoid a circular
    import between ``worldsim.nations`` and ``worldsim.ai.rnn``.
    """

    _LORA_R: int = 8
    _LORA_ALPHA: float = 16.0
    _LR_LORA: float = 3e-4

    def __init__(
        self,
        core_state_fn,
        *,
        lora_r: int = _LORA_R,
        lora_alpha: float = _LORA_ALPHA,
        lr_lora: float = _LR_LORA,
    ) -> None:
        self.core_state = core_state_fn
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha

        self.lora_A_core = nn.Parameter(torch.empty(lora_r, CORE_DIM, device=_DEVICE))
        self.lora_B_core = nn.Parameter(torch.zeros(HIDDEN_DIM, lora_r, device=_DEVICE))
        nn.init.kaiming_uniform_(self.lora_A_core, a=math.sqrt(5))
        with torch.no_grad():
            self._target_A_core = self.lora_A_core.data.clone()
            self._target_B_core = self.lora_B_core.data.clone()
        self._lr_lora = lr_lora
        self._core_optimizer = optim.Adam([self.lora_A_core, self.lora_B_core], lr=lr_lora)

        self.roles: Dict[str, UnifiedRoleView] = {}

    def sync_core_target(self) -> None:
        """Hard-copy the online core LoRA into its frozen target."""
        with torch.no_grad():
            self._target_A_core.copy_(self.lora_A_core.data)
            self._target_B_core.copy_(self.lora_B_core.data)

    def mutate_core(self, sigma: float = 0.02) -> None:
        with torch.no_grad():
            for p in (self.lora_A_core, self.lora_B_core):
                p.data.add_(torch.randn_like(p.data) * sigma)
        self.sync_core_target()

    def reset_core_optimizer(self) -> None:
        self._core_optimizer = optim.Adam(
            [self.lora_A_core, self.lora_B_core], lr=self._lr_lora
        )

    def reset_runtime_state(self) -> None:
        """Zero the transient GRU hidden state/rolling buffer for every role
        view this nation owns — see
        :meth:`UnifiedRoleView.reset_runtime_state`. Call before persisting
        (see ``worldsim.ai.persistence``)."""
        for view in self.roles.values():
            view.reset_runtime_state()

    def save_core(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "lora_A_core": self.lora_A_core.data,
                "lora_B_core": self.lora_B_core.data,
                "target_A_core": self._target_A_core,
                "target_B_core": self._target_B_core,
            },
            path,
        )

    def load_core(self, path: Path) -> None:
        path = Path(path)
        if not path.exists():
            return
        sd = torch.load(path, map_location=_DEVICE, weights_only=True)
        if "lora_A_core" in sd and sd["lora_A_core"].shape == self.lora_A_core.shape:
            self.lora_A_core.data.copy_(sd["lora_A_core"].to(_DEVICE))
        if "lora_B_core" in sd and sd["lora_B_core"].shape == self.lora_B_core.shape:
            self.lora_B_core.data.copy_(sd["lora_B_core"].to(_DEVICE))
        if "target_A_core" in sd:
            self._target_A_core = sd["target_A_core"].to(_DEVICE).clone()
        if "target_B_core" in sd:
            self._target_B_core = sd["target_B_core"].to(_DEVICE).clone()

    # ``save_table``/``load_table`` aliases so the bank fits the same
    # per-nation persistence convention as every role view (see
    # ``worldsim.ai.persistence._collect_ai_controllers``, role key "core").
    def save_table(self, path: Path) -> None:
        self.save_core(Path(path).with_suffix(".lora.pt"))

    def load_table(self, path: Path) -> None:
        self.load_core(Path(path).with_suffix(".lora.pt"))
