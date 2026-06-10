"""Simple dense perceptron controller (CPU-only).

The GPU-accelerated implementation lives in :mod:`worldsim.ai.gpu`.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence
import math
import random

_GELU_COEF = math.sqrt(2.0 / math.pi)
_GELU_KAPPA = 0.044715


class SimplePerceptron:
    """Light-weight dense controller built around GeLU activations.

    This variant intentionally uses only CPU/Python primitives.  The
    GPU-accelerated implementation lives in :mod:`worldsim.ai.gpu`.
    """

    def __init__(
        self,
        n_inputs: int,
        *,
        n_outputs: int = 3,
        hebbian_lr: float = 0.02,
        weight_clip: float = 3.5,
    ) -> None:
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self._weights: List[List[float]] = [
            [random.uniform(-0.5, 0.5) for _ in range(n_inputs)]
            for _ in range(n_outputs)
        ]
        self._bias: List[float] = [0.0] * n_outputs
        self.hebbian_lr = max(0.0, hebbian_lr)
        self.weight_clip = max(0.5, weight_clip)

    # Small helpers -----------------------------------------------------
    def _prepare_inputs(self, inputs: Sequence[float]) -> List[float]:
        vec = list(inputs)[: self.n_inputs]
        if len(vec) < self.n_inputs:
            vec.extend([0.0] * (self.n_inputs - len(vec)))
        return vec

    def _clip_weights(self) -> None:
        limit = self.weight_clip
        for row in self._weights:
            for idx, value in enumerate(row):
                row[idx] = max(-limit, min(limit, value))
        for idx, value in enumerate(self._bias):
            self._bias[idx] = max(-limit, min(limit, value))

    def _compute_linear(self, inputs: Sequence[float]):
        vec = self._prepare_inputs(inputs)
        outputs: List[float] = []
        for weights, bias in zip(self._weights, self._bias):
            outputs.append(sum(w * x for w, x in zip(weights, vec)) + bias)
        return outputs, vec

    # Activation -------------------------------------------------------
    def _gelu_scalar(self, value: float) -> float:
        inner = _GELU_COEF * (value + _GELU_KAPPA * value**3)
        return 0.5 * value * (1.0 + math.tanh(inner))

    def _gelu_derivative_scalar(self, value: float) -> float:
        inner = _GELU_COEF * (value + _GELU_KAPPA * value**3)
        tanh_inner = math.tanh(inner)
        sech_sq = 1.0 - tanh_inner**2
        return 0.5 * (1.0 + tanh_inner) + 0.5 * value * sech_sq * _GELU_COEF * (
            1.0 + 3.0 * _GELU_KAPPA * value**2
        )

    # Properties --------------------------------------------------------
    @property
    def weights(self) -> List[List[float]]:
        return [list(row) for row in self._weights]

    @weights.setter
    def weights(self, value: Sequence[Sequence[float]]) -> None:
        self._weights = [list(row) for row in value]

    @property
    def bias(self) -> List[float]:
        return list(self._bias)

    @bias.setter
    def bias(self, value: Sequence[float]) -> None:
        self._bias = list(value)

    # Public API --------------------------------------------------------
    def predict(self, inputs: Sequence[float]) -> List[float]:
        linear, _ = self._compute_linear(inputs)
        return [self._gelu_scalar(val) for val in linear]

    def predict_prob(self, inputs: Sequence[float]) -> List[float]:
        linear, _ = self._compute_linear(inputs)
        if not linear:
            return []
        activated = [self._gelu_scalar(val) for val in linear]
        max_logit = max(activated)
        exp_vals = [math.exp(value - max_logit) for value in activated]
        total = sum(exp_vals) or 1.0
        return [val / total for val in exp_vals]

    def train(
        self,
        batch_inputs: Iterable[Sequence[float]],
        batch_targets: Iterable[Sequence[float]],
        lr: float = 0.1,
    ) -> None:
        paired_inputs: List[Sequence[float]] = []
        paired_targets: List[List[float]] = []
        for inputs, targets in zip(batch_inputs, batch_targets):
            target_list = list(targets)
            if len(target_list) != self.n_outputs:
                continue
            paired_inputs.append(inputs)
            paired_targets.append(target_list)

        if not paired_inputs:
            return

        for inputs, target_list in zip(paired_inputs, paired_targets):
            linear, vec = self._compute_linear(inputs)
            activated = [self._gelu_scalar(val) for val in linear]
            if not activated:
                continue
            max_logit = max(activated)
            exp_vals = [math.exp(value - max_logit) for value in activated]
            total = sum(exp_vals) or 1.0
            probs = [val / total for val in exp_vals]
            errors = [t - p for t, p in zip(target_list, probs)]
            derivatives = [self._gelu_derivative_scalar(val) for val in linear]
            grads = [err * deriv for err, deriv in zip(errors, derivatives)]
            for out_idx, grad_val in enumerate(grads):
                for inp_idx, value in enumerate(vec):
                    self._weights[out_idx][inp_idx] += lr * grad_val * value
                self._bias[out_idx] += lr * grad_val
                if self.hebbian_lr:
                    hebb = probs[out_idx]
                    for inp_idx, value in enumerate(vec):
                        self._weights[out_idx][inp_idx] += self.hebbian_lr * hebb * value
                    self._bias[out_idx] += self.hebbian_lr * hebb
            if self.hebbian_lr:
                self._clip_weights()
