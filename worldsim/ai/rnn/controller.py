"""Per-nation GRU controller with LoRA specialisation.

See the package docstring of :mod:`worldsim.ai.rnn` for the architecture
overview (rolling input buffer, LoRA adapters, Double DQN training and the
MetaGA integration).
"""
from __future__ import annotations

import math
import random
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .base import BUFFER_SIZE, _DEVICE, get_role_model


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
    callers (e.g. :class:`~worldsim.nations.nation.Nation`) work without
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
