"""Layered policy optimised with Nelder-Mead search.

:class:`NelderMeadPolicy` is the base class for the role facades in
:mod:`worldsim.ai.roles`.  It wraps a :class:`~worldsim.ai.networks.LayeredNetwork`
with an epsilon-greedy action interface, a small replay memory and YAML
persistence of the learned per-neuron scales.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Deque, Iterable, List, Optional, Sequence, Tuple
import math
import random

import yaml

from .nelder_mead import nelder_mead_minimise
from .networks import LayeredNetwork


class NelderMeadPolicy:
    """Layered policy optimised using Nelder-Mead search."""

    def __init__(
        self,
        n_inputs: int,
        n_actions: int,
        *,
        hidden_layers: Sequence[int] = (18, 14, 12, 9, 6, 4),
        table_path: str | Path | None = None,
        epsilon: float = 0.1,
        gamma: float = 0.95,
        memory_size: int = 24,
        step_scale: float = 0.15,
        max_iter: int = 35,
        tol: float = 1e-3,
        seed: Optional[int] = None,
        hebbian_lr: float = 0.01,
        hebbian_clip: float = 4.5,
    ) -> None:
        self.n_inputs = n_inputs
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.gamma = gamma
        self.hidden_layers = tuple(int(v) for v in hidden_layers)
        self.step_scale = step_scale
        self.max_iter = max_iter
        self.tol = tol
        self._rng = random.Random(seed)
        self.hebbian_lr = max(0.0, hebbian_lr)
        self.hebbian_clip = max(1.0, hebbian_clip)
        memory_capacity = max(1, int(memory_size))
        self.memory_limit = memory_capacity
        self.memory: Deque[Tuple[List[float], List[float]]] = deque(maxlen=memory_capacity)
        self.network = LayeredNetwork(
            n_inputs,
            [*self.hidden_layers, n_actions],
            seed=self._rng.randint(0, 2**31 - 1),
        )
        self.scale_count = self._expected_scale_length()
        self.scales: List[float] = [1.0] * self.scale_count
        # Throttle expensive Nelder-Mead optimization: update memory every step
        # but only run the search every _optimise_every steps.
        self._train_step: int = 0
        self._optimise_every: int = 8
        self.table_path = Path(table_path) if table_path else None
        if self.table_path:
            self.load_table(self.table_path)

    # Helpers -----------------------------------------------------------
    def _prepare_state(self, state: Sequence[float]) -> List[float]:
        vec = list(state)[: self.n_inputs]
        if len(vec) < self.n_inputs:
            vec.extend([0.0] * (self.n_inputs - len(vec)))
        return vec

    def _expected_scale_length(self) -> int:
        return sum(len(layer) for layer in self.network.weights)

    def _loss(self, scales: Sequence[float], samples: Iterable[Tuple[List[float], List[float]]]) -> float:
        total_error = 0.0
        count = 0
        penalty = 0.0
        for scale in scales:
            if not math.isfinite(scale):
                return float("inf")
            if abs(scale) > 6.0:
                penalty += (abs(scale) - 6.0) ** 2
        for state, target in samples:
            prediction = self.network.forward(state, scales)
            if len(prediction) != len(target):
                continue
            err = 0.0
            for pred, goal in zip(prediction, target):
                diff = pred - goal
                err += diff * diff
            total_error += err / max(1, self.n_actions)
            count += 1
        if count == 0:
            return 0.0
        mse = total_error / count
        regulariser = 1e-3 * sum((scale - 1.0) ** 2 for scale in scales)
        return mse + regulariser + 0.05 * penalty

    def _optimise(self) -> None:
        if not self.memory:
            return

        samples = list(self.memory)
        initial = list(self.scales)

        def objective(vector: Sequence[float]) -> float:
            return self._loss(vector, samples)

        result = nelder_mead_minimise(
            objective,
            initial,
            step=self.step_scale,
            max_iter=self.max_iter,
            tol=self.tol,
        )
        scales = [
            max(0.05, min(6.0, float(value))) if math.isfinite(value) else 1.0
            for value in result.point
        ]
        expected = self._expected_scale_length()
        if len(scales) < expected:
            scales.extend([1.0] * (expected - len(scales)))
        elif len(scales) > expected:
            scales = scales[:expected]
        self.scales = scales
        self._persist_scales()

    def _persist_scales(self) -> None:
        if not self.table_path:
            return
        try:
            self.save_table()
        except OSError:
            if __debug__:
                pass

    # Public API --------------------------------------------------------
    def predict(self, state: Sequence[float]) -> List[float]:
        prepared = self._prepare_state(state)
        outputs = self.network.forward(prepared, self.scales)
        return [float(value) for value in outputs]

    def choose_action(self, state: Sequence[float], valid_mask: List[bool] | None = None) -> int:
        if self.n_actions <= 0:
            return 0
        values = self.predict(state)
        if not values:
            return 0
        valid_actions = min(self.n_actions, len(values))
        if valid_actions <= 0:
            return 0
        if self._rng.random() < self.epsilon:
            # Random exploration only among valid actions
            if valid_mask:
                valid_indices = [i for i in range(valid_actions) if valid_mask[i]]
                return self._rng.choice(valid_indices) if valid_indices else 0
            return self._rng.randrange(valid_actions)
    
        best_idx = 0
        best_val = float('-inf')
        for idx in range(valid_actions):
            if valid_mask and not valid_mask[idx]:
                continue  # skip invalid actions
            if values[idx] > best_val:
                best_idx = idx
                best_val = values[idx]
        return min(best_idx, self.n_actions - 1)

    def train(
        self,
        state: Sequence[float],
        action: int,
        reward: float,
        next_state: Sequence[float],
    ) -> None:
        if not (0 <= action < self.n_actions):
            return
        state_vec = self._prepare_state(state)
        next_vec = self._prepare_state(next_state)
        forward_result = self.network.forward(state_vec, self.scales, return_trace=True)
        if isinstance(forward_result, tuple):
            current, trace = forward_result
        else:
            current = list(forward_result)
            trace = []
        if not current:
            return

        if len(current) < self.n_actions:
            current.extend([0.0] * (self.n_actions - len(current)))
        elif len(current) > self.n_actions:
            current = current[: self.n_actions]

        next_values = self.network.forward(next_vec, self.scales)
        if isinstance(next_values, tuple):  # safety for legacy tables
            future_values = list(next_values[0])
        else:
            future_values = list(next_values)
        if len(future_values) < len(current):
            future_values.extend([0.0] * (len(current) - len(future_values)))
        elif len(future_values) > len(current):
            future_values = future_values[: len(current)]

        target = current
        clamped_action = min(max(action, 0), len(target) - 1)
        future_best = max(future_values) if future_values else 0.0
        target[clamped_action] = reward + self.gamma * future_best
        self.memory.append((state_vec, target))
        if trace and self.hebbian_lr:
            self.network.apply_hebbian(trace, self.hebbian_lr, weight_clip=self.hebbian_clip)
        self._train_step += 1
        if self._train_step % self._optimise_every == 0:
            self._optimise()

    # Persistence -------------------------------------------------------
    def save_table(self, path: str | Path | None = None) -> None:
        target = Path(path) if path else self.table_path
        if target is None:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        rng_version, rng_state, rng_gauss = self._rng.getstate()
        payload = {
            "type": "nelder_mead_policy",
            "n_inputs": self.n_inputs,
            "n_actions": self.n_actions,
            "epsilon": self.epsilon,
            "gamma": self.gamma,
            "hidden_layers": list(self.hidden_layers),
            "scales": list(self.scales),
            "network": self.network.to_dict(),
            "step_scale": self.step_scale,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "memory_size": self.memory.maxlen,
            "hebbian_lr": self.hebbian_lr,
            "hebbian_clip": self.hebbian_clip,
            "memory": [
                {"state": list(state), "target": list(target)}
                for state, target in self.memory
            ],
            "rng_state": {
                "version": int(rng_version),
                "state": list(rng_state),
                "gauss": rng_gauss,
            },
        }
        with target.open("w", encoding="utf8") as fh:
            yaml.safe_dump(payload, fh)
        self.table_path = target

    def load_table(self, path: str | Path) -> None:
        target = Path(path)
        if not target.exists():
            return
        with target.open("r", encoding="utf8") as fh:
            payload = yaml.safe_load(fh) or {}
        network_data = payload.get("network")
        if isinstance(network_data, dict):
            self.network = LayeredNetwork.from_dict(network_data)
            self.n_inputs = self.network.n_inputs
            self.n_actions = len(self.network.weights[-1]) if self.network.weights else 0
            self.hidden_layers = tuple(int(size) for size in self.network.layer_sizes[1:-1])
        hidden_layers = payload.get("hidden_layers")
        if isinstance(hidden_layers, list) and hidden_layers:
            self.hidden_layers = tuple(int(v) for v in hidden_layers)
        expected = self._expected_scale_length()
        scales = payload.get("scales")
        if isinstance(scales, list) and scales:
            parsed = [float(value) for value in scales]
            if len(parsed) < expected:
                parsed.extend([1.0] * (expected - len(parsed)))
            elif len(parsed) > expected:
                parsed = parsed[:expected]
            self.scales = parsed
        else:
            self.scales = [1.0] * expected
        self.scale_count = len(self.scales)
        epsilon = payload.get("epsilon")
        gamma = payload.get("gamma")
        if isinstance(epsilon, (int, float)):
            self.epsilon = float(epsilon)
        if isinstance(gamma, (int, float)):
            self.gamma = float(gamma)
        step_scale = payload.get("step_scale")
        max_iter = payload.get("max_iter")
        tol = payload.get("tol")
        if isinstance(step_scale, (int, float)):
            self.step_scale = float(step_scale)
        if isinstance(max_iter, int) and max_iter > 0:
            self.max_iter = max_iter
        if isinstance(tol, (int, float)) and tol > 0:
            self.tol = float(tol)
        hebbian_lr = payload.get("hebbian_lr")
        if isinstance(hebbian_lr, (int, float)):
            self.hebbian_lr = max(0.0, float(hebbian_lr))
        hebbian_clip = payload.get("hebbian_clip")
        if isinstance(hebbian_clip, (int, float)):
            self.hebbian_clip = max(1.0, float(hebbian_clip))
        memory_size = payload.get("memory_size")
        if isinstance(memory_size, int) and memory_size > 0:
            self.memory_limit = memory_size
            self.memory = deque(maxlen=memory_size)
        else:
            self.memory = deque(maxlen=self.memory_limit)
        self.memory.clear()
        memory_payload = payload.get("memory")
        if isinstance(memory_payload, list):
            for entry in memory_payload:
                if not isinstance(entry, dict):
                    continue
                state = entry.get("state")
                target = entry.get("target")
                if not isinstance(state, list) or not isinstance(target, list):
                    continue
                prepared_state = self._prepare_state(state)
                target_vec = list(target)
                if len(target_vec) < self.n_actions:
                    target_vec.extend([0.0] * (self.n_actions - len(target_vec)))
                elif len(target_vec) > self.n_actions:
                    target_vec = target_vec[: self.n_actions]
                self.memory.append((prepared_state, target_vec))
        rng_payload = payload.get("rng_state")
        if isinstance(rng_payload, dict):
            version = rng_payload.get("version", 3)
            state_seq = rng_payload.get("state")
            gauss = rng_payload.get("gauss")
            if isinstance(state_seq, list):
                try:
                    state_tuple = tuple(int(v) for v in state_seq)
                    self._rng.setstate((int(version), state_tuple, gauss))
                except (TypeError, ValueError):
                    pass
        self.table_path = target

