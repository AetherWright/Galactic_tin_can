"""PyTorch RNN AI backend for WorldSim nation controllers.

Architecture
------------
One GRU-based *base model* is shared across all nations for each AI *role*
(war, civilian, project, diplomacy, research, doctrine, fleet).  Each nation
additionally owns a lightweight *LoRA adapter* per role that specialises the
shared backbone without touching its weights.

Rolling input buffer
    Every controller keeps a ``deque`` of the last :data:`BUFFER_SIZE`
    (= 5) input vectors.  At each forward pass the full buffer is fed to the
    GRU as a sequence, giving the model temporal context across simulation
    steps.

LoRA (Low-Rank Adaptation)
    The adapter modifies the input-projection and output-projection layers of
    the base GRU:

        eff_W_in  = W_in  + (B_in  @ A_in)  * (alpha / r)
        eff_W_out = W_out + (B_out @ A_out) * (alpha / r)

    ``A`` matrices are kaiming-uniform-initialised; ``B`` matrices start at
    zero, so the adapter has zero effect at initialisation and only diverges
    from the base model as training proceeds.

Training
    *LoRA*: Updated on every :meth:`RNNController.train` call using TD(0) and
    a per-nation Adam optimiser.  The backward pass is serialised through a
    per-role lock so concurrent nation threads do not corrupt shared base
    model parameter gradients.

    *Base model*: Each train call stores the experience in a replay buffer
    owned by the :class:`RoleBaseModel`.  Call
    :func:`step_all_base_models` from the main (non-threaded) simulation
    loop once per century to run gradient updates over the accumulated batch.

Public API (matches the old NelderMeadPolicy surface)
    ``predict(state) -> List[float]``
    ``predict_prob(state) -> List[float]``
    ``choose_action(state, valid_mask=None) -> int``
    ``train(state, action, reward, next_state) -> None``
    ``save_table(path) / load_table(path)``
"""
from __future__ import annotations

import math
import random
import threading
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple

_TORCH_AVAILABLE: bool = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
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

# Number of past input states kept in the rolling buffer.
BUFFER_SIZE: int = 5

# Doctrine and fleet-state labels (mirror military_ai.py to avoid circular import)
_DOCTRINE_LIST: List[str] = [
    "offensive", "defensive", "economic", "strategic_reserve", "total_war"
]
_FLEET_STATE_LIST: List[str] = [
    "patrol", "assault", "defend", "escort", "retreat", "reposition", "colonize"
]


# ---------------------------------------------------------------------------
# PyTorch classes — only defined when torch is importable
# ---------------------------------------------------------------------------

# Role → per-nation RewardGA key.  Fleet and doctrine use the civilian GA
# as a proxy because they have no dedicated reward tracker.
_ROLE_GA_KEY: dict = {
    "civilian":  "civilian",
    "project":   "projects",
    "diplomacy": "diplomacy",
    "research":  "research",
    "war":       "civilian",   # proxy
    "doctrine":  "civilian",   # proxy
    "fleet":     "civilian",   # proxy
}


if _TORCH_AVAILABLE:

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
        _REPLAY_CAP: int   = 8_000
        _BATCH_SIZE: int   = 64
        _GAMMA:      float = 0.95

        def __init__(
            self, input_dim: int, hidden_dim: int, n_outputs: int
        ) -> None:
            self.model     = _BaseGRUModel(input_dim, hidden_dim, n_outputs)
            self._optimizer = optim.Adam(self.model.parameters(), lr=self._BASE_LR)
            # Lock serialises backward passes from concurrent nation threads.
            self._lock     = threading.Lock()
            # Replay buffer: each entry is (cur_buf, action, reward, next_buf)
            self._replay: Deque[
                Tuple[List[List[float]], int, float, List[List[float]]]
            ] = deque(maxlen=self._REPLAY_CAP)
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
            if len(replay_snap) < self._BATCH_SIZE:
                return

            h0 = torch.zeros(1, 1, self.model.hidden_dim, device=_DEVICE)

            for _ in range(n_steps):
                batch = random.sample(
                    replay_snap, min(self._BATCH_SIZE, len(replay_snap))
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


    def sync_all_meta_ga(nations: object) -> None:
        """Push MetaGA fitness state into every nation's RNN controllers.

        Call this from the simulation's century loop *after*
        ``ga.step(score)`` has been called for all nations and *before* the
        per-fifth simulation steps begin.

        For each nation the appropriate :class:`~worldsim.meta_ga.RewardGA`
        instance (keyed by :data:`_ROLE_GA_KEY`) is passed to the
        controller's :meth:`~RNNController.sync_meta_ga` method so that
        epsilon and reward-scale reflect the current genome's fitness.
        """
        _attr_role_ga: list = [
            ("civilian_ai",  "civilian"),
            ("project_ai",   "projects"),
            ("diplomacy_ai", "diplomacy"),
            ("research_ai",  "research"),
            ("military_ai",  "civilian"),   # proxy
            ("doctrine_ai",  "civilian"),   # proxy
        ]
        for nation in nations.values():                          # type: ignore[attr-defined]
            for attr, ga_key in _attr_role_ga:
                ctrl = getattr(nation, attr, None)
                if not hasattr(ctrl, "sync_meta_ga"):
                    continue
                ga = nation.reward_ga.get(ga_key)               # type: ignore[attr-defined]
                if ga is not None:
                    ctrl.sync_meta_ga(ga)


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

    # -----------------------------------------------------------------------
    # Per-nation RNN controller with LoRA
    # -----------------------------------------------------------------------

    class RNNController:
        """Nation-specific GRU controller with LoRA specialisation.

        The controller wraps a :class:`RoleBaseModel` and adds per-nation
        LoRA parameters and a :data:`BUFFER_SIZE`-deep rolling input buffer.

        Training algorithm
        ------------------
        **Double DQN** with a frozen *target network*:

        1. The *online* (main) network selects the greedy next action from
           the next-state Q-values.
        2. The *target* network evaluates that action's Q-value, producing
           a less biased bootstrap target.
        3. The target LoRA is hard-copied from the online LoRA every
           :attr:`TARGET_UPDATE_FREQ` training steps.

        MetaGA integration
        ------------------
        :meth:`sync_meta_ga` accepts the nation's :class:`~worldsim.meta_ga.RewardGA`
        and uses its active genome fitness to:

        * **Adaptive ε-greedy**: ``ε = base_ε · exp(−max(fitness, 0) / 80)``
          — well-performing genomes exploit their learned policy more.
        * **Reward scaling**: the TD target is multiplied by a factor that
          grows with fitness, reinforcing strategies that are objectively
          working at the simulation level.
        * **Genome-change detection**: when the GA promotes a new genome
          (after ``evolve_meta``), the Adam optimiser is reset so momentum
          buffers from the previous genome's experience do not bias the new
          reward weighting.

        :meth:`mutate_lora` applies Gaussian noise to all LoRA tensors and
        is called by the nation's ``evolve_meta`` so the behaviour space is
        explored alongside the new reward landscape.

        Parameters
        ----------
        role:
            Identifies which :class:`RoleBaseModel` to share.
        n_inputs:
            Length of the input feature vector.
        n_outputs:
            Number of action scores to produce.
        hidden_dim:
            GRU hidden-state width.
        lora_r:
            LoRA rank (default 8).
        lora_alpha:
            LoRA scaling factor (default 16.0).
        epsilon:
            Base ε-greedy exploration rate (modulated by MetaGA).
        gamma:
            TD discount factor.
        lr_lora:
            Adam learning rate for LoRA parameters.
        seed:
            Optional RNG seed for reproducible exploration.
        """

        _LORA_R:            int   = 8
        _LORA_ALPHA:        float = 16.0
        _GAMMA:             float = 0.95
        _EPSILON:           float = 0.10
        _LR_LORA:           float = 3e-4
        # Double DQN: hard-copy online → target every N train calls
        TARGET_UPDATE_FREQ: int   = 16
        # MetaGA: fitness normalisation constant (≈ max expected per-century score)
        _FITNESS_SCALE:     float = 80.0
        # Floor on epsilon even for very high-fitness genomes
        _EPSILON_FLOOR:     float = 0.02

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
            self._base_epsilon = epsilon   # reference value; GA modulates from here
            self.gamma      = gamma
            self._rng       = random.Random(seed)

            # Shared base model (one per role × input-dim × output-dim combo)
            self.base = get_role_model(role, n_inputs, hidden_dim, n_outputs)

            # LoRA parameters — nation-specific (online network)
            # Input projection:  in=n_inputs, out=hidden_dim
            self.lora_A_in  = nn.Parameter(torch.empty(lora_r, n_inputs, device=_DEVICE))
            self.lora_B_in  = nn.Parameter(torch.zeros(hidden_dim, lora_r, device=_DEVICE))
            # Output projection: in=hidden_dim, out=n_outputs
            self.lora_A_out = nn.Parameter(torch.empty(lora_r, hidden_dim, device=_DEVICE))
            self.lora_B_out = nn.Parameter(torch.zeros(n_outputs, lora_r, device=_DEVICE))

            # Kaiming-uniform A; B=0 → ΔW=0 at init
            nn.init.kaiming_uniform_(self.lora_A_in,  a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_A_out, a=math.sqrt(5))

            self._lora_optimizer = optim.Adam(
                [self.lora_A_in, self.lora_B_in,
                 self.lora_A_out, self.lora_B_out],
                lr=lr_lora,
            )
            self._lr_lora = lr_lora

            # Double DQN target network — frozen copies of online LoRA
            with torch.no_grad():
                self._target_A_in  = self.lora_A_in.data.clone()
                self._target_B_in  = self.lora_B_in.data.clone()
                self._target_A_out = self.lora_A_out.data.clone()
                self._target_B_out = self.lora_B_out.data.clone()
            self._train_steps: int = 0

            # MetaGA integration state
            self._genome_id:    int   = -1    # last-known active genome index
            self._reward_scale: float = 1.0   # GA-fitness reward amplifier

            # Per-nation recurrent state (detached across turns — truncated BPTT)
            self._h: torch.Tensor = torch.zeros(1, 1, hidden_dim, device=_DEVICE)

            # Rolling input buffer, pre-filled with zeros
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

        def _forward_tensor(
            self,
            x_seq: "torch.Tensor",  # [1, T, n_inputs]
            h:     "torch.Tensor",  # [1, 1, hidden_dim]
        ) -> "torch.Tensor":         # [1, n_outputs]
            """Forward pass with LoRA applied to input and output projections."""
            base  = self.base.model
            scale = self.lora_alpha / self.lora_r
            T     = x_seq.shape[1]
            x_flat = x_seq.reshape(T, self.n_inputs)  # [T, n_inputs]

            # Input projection: base + LoRA
            proj_in   = F.linear(x_flat, base.input_proj.weight,
                                  base.input_proj.bias)
            lora_in_  = (x_flat @ self.lora_A_in.T) @ self.lora_B_in.T * scale
            x_proj    = (proj_in + lora_in_).unsqueeze(0)  # [1, T, hidden_dim]

            # GRU (base weights only)
            gru_out, h_new = base.gru(x_proj, h)           # [1, T, hidden_dim]
            last = gru_out[:, -1]                            # [1, hidden_dim]

            # Output projection: base + LoRA
            proj_out  = F.linear(last, base.output_proj.weight,
                                  base.output_proj.bias)
            lora_out_ = (last @ self.lora_A_out.T) @ self.lora_B_out.T * scale
            scores    = proj_out + lora_out_                 # [1, n_outputs]

            return scores

        def _forward_tensor_target(
            self,
            x_seq: "torch.Tensor",  # [1, T, n_inputs]
            h:     "torch.Tensor",  # [1, 1, hidden_dim]
        ) -> "torch.Tensor":         # [1, n_outputs]
            """Forward pass using the frozen TARGET LoRA tensors.

            Used by Double DQN to evaluate the Q-value of the next-state
            greedy action without the moving-target problem.
            """
            base  = self.base.model
            scale = self.lora_alpha / self.lora_r
            T     = x_seq.shape[1]
            x_flat = x_seq.reshape(T, self.n_inputs)

            proj_in   = F.linear(x_flat, base.input_proj.weight,
                                  base.input_proj.bias)
            lora_in_  = (x_flat @ self._target_A_in.T) @ self._target_B_in.T * scale
            x_proj    = (proj_in + lora_in_).unsqueeze(0)

            gru_out, _ = base.gru(x_proj, h)
            last = gru_out[:, -1]

            proj_out  = F.linear(last, base.output_proj.weight,
                                  base.output_proj.bias)
            lora_out_ = (last @ self._target_A_out.T) @ self._target_B_out.T * scale
            return proj_out + lora_out_

        def _update_target_network(self) -> None:
            """Hard-copy online LoRA → frozen target LoRA."""
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
            """Forward pass; returns one raw score per action.

            Appends *state* to the rolling buffer, runs the GRU over the
            buffer, and advances the hidden state.
            """
            vec = self._pad(state)
            self._buffer.append(vec)
            seq = self._buf_to_tensor(self._buffer)
            with torch.no_grad():
                scores = self._forward_tensor(seq, self._h)
                # advance hidden state through base GRU (no LoRA needed here)
                x_flat = seq.reshape(seq.shape[1], self.n_inputs)
                proj   = F.linear(
                    x_flat,
                    self.base.model.input_proj.weight,
                    self.base.model.input_proj.bias,
                )
                _, h_new = self.base.model.gru(
                    proj.unsqueeze(0), self._h
                )
            self._h = h_new.detach()
            return scores[0].tolist()

        def predict_prob(self, state: Sequence[float]) -> List[float]:
            """Softmax probabilities over actions."""
            raw = self.predict(state)
            t = torch.tensor(raw)
            return torch.softmax(t, dim=0).tolist()

        def choose_action(
            self,
            state:      Sequence[float],
            valid_mask: Optional[List[bool]] = None,
        ) -> int:
            """ε-greedy action selection."""
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
            """Double DQN update for LoRA parameters with MetaGA reward scaling.

            Double DQN de-couples action *selection* (online network) from
            action *evaluation* (target network), reducing the overestimation
            bias of vanilla DQN::

                best_a  = argmax_a  Q_online(s', a)      # main net picks action
                target  = r * scale + γ · Q_target(s', best_a)  # target net evaluates

            The target LoRA is hard-copied from the online LoRA every
            :attr:`TARGET_UPDATE_FREQ` train calls.  The MetaGA
            ``_reward_scale`` amplifies the reward for genomes that are
            performing well at the simulation level, reinforcing successful
            strategies more strongly.

            The backward pass is serialised through the base model's lock so
            concurrent nation-turn threads cannot corrupt shared parameter
            gradients.  After the LoRA step, base model gradients are zeroed;
            the base model is trained separately via :func:`step_all_base_models`.

            The experience (with scaled reward) is also deposited in the
            role's replay buffer.
            """
            if not (0 <= action < self.n_outputs):
                return

            # MetaGA reward scaling — amplify signal for well-performing genomes
            scaled_reward = float(reward) * self._reward_scale

            next_vec  = self._pad(next_state)
            cur_buf   = list(self._buffer)       # buffer has *state* already appended
            next_buf_deq: Deque[List[float]] = deque(self._buffer, maxlen=BUFFER_SIZE)
            next_buf_deq.append(next_vec)
            next_buf = list(next_buf_deq)

            cur_seq  = torch.tensor(cur_buf,  dtype=torch.float32, device=_DEVICE).unsqueeze(0)
            next_seq = torch.tensor(next_buf, dtype=torch.float32, device=_DEVICE).unsqueeze(0)
            h_ref    = self._h  # detached; on _DEVICE

            # Serialise backward through the shared base model
            with self.base._lock:
                # --- Double DQN target computation ---
                with torch.no_grad():
                    # 1. Online network selects the greedy next action
                    next_q_online   = self._forward_tensor(next_seq, h_ref)
                    best_next_act   = int(next_q_online.argmax(dim=1).item())
                    # 2. Target network evaluates that action (no overestimation)
                    next_q_target   = self._forward_tensor_target(next_seq, h_ref)
                    target_val      = scaled_reward + self.gamma * float(
                        next_q_target[0, best_next_act]
                    )

                # Current Q values (grad flows through LoRA params)
                cur_scores = self._forward_tensor(cur_seq, h_ref)
                target     = cur_scores.detach().clone()
                target[0, action] = target_val
                loss = F.mse_loss(cur_scores, target)

                # Update LoRA
                self._lora_optimizer.zero_grad()
                loss.backward()
                self._lora_optimizer.step()

                # Zero out gradients that landed on base model params
                # (base model is trained separately via replay)
                self.base.model.zero_grad()

            # Periodic hard-update of the target network
            self._train_steps += 1
            if self._train_steps % self.TARGET_UPDATE_FREQ == 0:
                self._update_target_network()

            # Deposit experience (scaled reward) in the role's replay buffer
            self.base.add_experience((cur_buf, action, scaled_reward, next_buf))

        # ------------------------------------------------------------------
        # MetaGA integration
        # ------------------------------------------------------------------

        def sync_meta_ga(self, reward_ga: object) -> None:
            """Synchronise controller hyper-parameters with the MetaGA state.

            Call once per century (after ``ga.step(score)`` has accumulated
            the century fitness) from :func:`sync_all_meta_ga`.

            Effects
            -------
            * **Adaptive ε**: ``ε = base_ε · exp(−max(fitness,0) / FITNESS_SCALE)``
              clipped to :attr:`_EPSILON_FLOOR`.  Successful genomes exploit
              their learned policy more; struggling genomes explore more.
            * **Reward scale**: ``scale = clamp(1 + fitness/200, 0.5, 2.0)``.
              Positive fitness amplifies the RL reward signal, reinforcing
              strategies that are genuinely working.
            * **Genome-change reset**: when the GA promotes a new active
              genome (i.e. ``reward_ga.active`` differs from the last-seen
              id), the Adam optimizer state is cleared so stale momentum
              does not bias the new reward landscape.
            """
            new_genome_id = reward_ga.active                            # type: ignore[attr-defined]
            fitness       = reward_ga.population[new_genome_id].fitness # type: ignore[attr-defined]

            # Detect genome change → reset optimizer so momentum from the
            # previous genome does not bias the new reward weighting.
            if new_genome_id != self._genome_id and self._genome_id != -1:
                self.reset_lora_optimizer()
            self._genome_id = new_genome_id

            # Fitness-adaptive exploration rate
            self.epsilon = max(
                self._EPSILON_FLOOR,
                self._base_epsilon * math.exp(-max(0.0, fitness) / self._FITNESS_SCALE),
            )

            # Fitness-adaptive reward scale  (clamped to [0.5, 2.0])
            self._reward_scale = max(0.5, min(2.0, 1.0 + fitness / 200.0))

        def mutate_lora(self, sigma: float = 0.02) -> None:
            """Apply Gaussian noise to all online LoRA tensors.

            Called by the nation's ``evolve_meta`` on collapse / rebirth so
            the LoRA behaviour space is explored in parallel with the GA's
            new reward weighting.

            The target network is synchronised after mutation so it starts
            from the same noisy baseline rather than lagging behind.
            """
            with torch.no_grad():
                for p in (self.lora_A_in, self.lora_B_in,
                           self.lora_A_out, self.lora_B_out):
                    p.data.add_(torch.randn_like(p.data) * sigma)
            # Sync target immediately so it doesn't lag behind the mutation
            self._update_target_network()

        def reset_lora_optimizer(self) -> None:
            """Reset the Adam optimizer, clearing momentum/variance buffers.

            Called automatically when the active GA genome changes, and can
            also be called manually to give the controller a fresh start.
            """
            self._lora_optimizer = optim.Adam(
                [self.lora_A_in, self.lora_B_in,
                 self.lora_A_out, self.lora_B_out],
                lr=self._lr_lora,
            )

        # ------------------------------------------------------------------
        # Persistence
        # ------------------------------------------------------------------

        def save_lora(self, path: Path) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "lora_A_in":       self.lora_A_in.data,
                    "lora_B_in":       self.lora_B_in.data,
                    "lora_A_out":      self.lora_A_out.data,
                    "lora_B_out":      self.lora_B_out.data,
                    "target_A_in":     self._target_A_in,
                    "target_B_in":     self._target_B_in,
                    "target_A_out":    self._target_A_out,
                    "target_B_out":    self._target_B_out,
                    "h":               self._h,
                    "train_steps":     self._train_steps,
                    "reward_scale":    self._reward_scale,
                    "genome_id":       self._genome_id,
                },
                path,
            )

        def load_lora(self, path: Path) -> None:
            if not path.exists():
                return
            sd = torch.load(path, map_location=_DEVICE, weights_only=True)
            if "lora_A_in"    in sd: self.lora_A_in.data.copy_(sd["lora_A_in"])
            if "lora_B_in"    in sd: self.lora_B_in.data.copy_(sd["lora_B_in"])
            if "lora_A_out"   in sd: self.lora_A_out.data.copy_(sd["lora_A_out"])
            if "lora_B_out"   in sd: self.lora_B_out.data.copy_(sd["lora_B_out"])
            if "target_A_in"  in sd: self._target_A_in.copy_(sd["target_A_in"])
            if "target_B_in"  in sd: self._target_B_in.copy_(sd["target_B_in"])
            if "target_A_out" in sd: self._target_A_out.copy_(sd["target_A_out"])
            if "target_B_out" in sd: self._target_B_out.copy_(sd["target_B_out"])
            if "h"            in sd: self._h = sd["h"]
            if "train_steps"  in sd: self._train_steps  = int(sd["train_steps"])
            if "reward_scale" in sd: self._reward_scale = float(sd["reward_scale"])
            if "genome_id"    in sd: self._genome_id    = int(sd["genome_id"])

    # -----------------------------------------------------------------------
    # Base class adding save_table / load_table compatibility shims
    # -----------------------------------------------------------------------

    class _RoleController(RNNController):
        """Mixin that exposes ``save_table`` / ``load_table`` so existing
        callers (e.g. :class:`~worldsim.models.nation.Nation`) work without
        change.
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
            lora_path = target.with_suffix(".lora.pt")
            self.save_lora(lora_path)
            self.table_path = target

        def load_table(self, path: str | Path) -> None:
            lora_path = Path(path).with_suffix(".lora.pt")
            self.load_lora(lora_path)
            self.table_path = Path(path)

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
            self._lora_optimizer = optim.Adam(
                [self.lora_A_in, self.lora_B_in,
                 self.lora_A_out, self.lora_B_out],
                lr=self._LR_LORA,
            )
            self._buffer = deque([[0.0] * new_n] * BUFFER_SIZE, maxlen=BUFFER_SIZE)
            self._h = torch.zeros(1, 1, self.hidden_dim, device=_DEVICE)

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
        :class:`~worldsim.models.military_ai.DoctrineAI`.

        Provides the same ``choose_doctrine`` / ``train`` interface used by
        :func:`~worldsim.models.military_ai.issue_doctrine`.
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
            from .models.military_ai import DoctrineAI as _DoctrineAI
            return _DoctrineAI.build_state(nation, nations)  # type: ignore[arg-type]


    class TorchFleetController(_RoleController):
        """RNN+LoRA replacement for
        :class:`~worldsim.models.military_ai.LayeredNetworkFleetController`.
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

else:
    # -----------------------------------------------------------------------
    # Stubs — imported safely when torch is absent
    # -----------------------------------------------------------------------
    RNNController          = None   # type: ignore[assignment,misc]
    RoleBaseModel          = None   # type: ignore[assignment,misc]
    TorchWarAI             = None   # type: ignore[assignment,misc]
    TorchDomesticPolicyAI  = None   # type: ignore[assignment,misc]
    TorchCivilianOverseer  = None   # type: ignore[assignment,misc]
    TorchDepartmentAI      = None   # type: ignore[assignment,misc]
    TorchProjectAI         = None   # type: ignore[assignment,misc]
    TorchDiplomacyAI       = None   # type: ignore[assignment,misc]
    TorchResearchAI        = None   # type: ignore[assignment,misc]
    TorchDoctrineAI        = None   # type: ignore[assignment,misc]
    TorchFleetController   = None   # type: ignore[assignment,misc]

    def get_role_model(*_a: object, **_kw: object) -> None:   # type: ignore[return]
        return None

    def step_all_base_models(*_a: object, **_kw: object) -> None:
        pass

    def sync_all_meta_ga(*_a: object, **_kw: object) -> None:
        pass

    def save_all_base_models(*_a: object, **_kw: object) -> None:
        pass

    def load_all_base_models(*_a: object, **_kw: object) -> None:
        pass


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------

def rnn_available() -> bool:
    """Return ``True`` if PyTorch is installed and the RNN backend is ready."""
    return _TORCH_AVAILABLE
