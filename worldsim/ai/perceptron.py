"""Simple dense perceptron controller.

Runs vectorised on the active array backend (CuPy on GPU, otherwise NumPy)
and falls back to pure-Python lists when neither is available.  See
:mod:`worldsim.ai._backend` for the ``WORLDSIM_AI_BACKEND`` override.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence
import math
import random

from ._backend import ai_array_module

_GELU_COEF = math.sqrt(2.0 / math.pi)
_GELU_KAPPA = 0.044715


class SimplePerceptron:
    """Light-weight dense controller built around GeLU activations.

    The neuron model retains the public API of the historical perceptron while
    replacing the linear activation with a modern Gaussian Error Linear Unit
    (GeLU).  The maths runs vectorised on NumPy/CuPy when available and keeps
    a pure-Python path as last resort, while continuing to expose the simple
    Hebbian update rule used by the surrounding controllers.
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
        self._xp = ai_array_module()
        initial_weights = [
            [random.uniform(-0.5, 0.5) for _ in range(n_inputs)]
            for _ in range(n_outputs)
        ]
        initial_bias = [0.0] * n_outputs
        if self._xp is not None:
            xp = self._xp
            self._weights = xp.asarray(initial_weights, dtype=xp.float64)
            self._bias = xp.asarray(initial_bias, dtype=xp.float64)
        else:
            self._weights = initial_weights
            self._bias = initial_bias
        self.hebbian_lr = max(0.0, hebbian_lr)
        self.weight_clip = max(0.5, weight_clip)

    # Small helpers -----------------------------------------------------
    def _prepare_inputs(self, inputs: Sequence[float]) -> List[float]:
        vec = list(inputs)[: self.n_inputs]
        if len(vec) < self.n_inputs:
            vec.extend([0.0] * (self.n_inputs - len(vec)))
        if self._xp is not None:
            xp = self._xp
            return xp.asarray(vec, dtype=xp.float64)
        return vec

    def _clip_weights(self) -> None:
        limit = self.weight_clip
        if self._xp is not None:
            xp = self._xp
            xp.clip(self._weights, -limit, limit, out=self._weights)
            xp.clip(self._bias, -limit, limit, out=self._bias)
            return
        for row in self._weights:
            for idx, value in enumerate(row):
                row[idx] = max(-limit, min(limit, value))
        for idx, value in enumerate(self._bias):
            self._bias[idx] = max(-limit, min(limit, value))

    def _compute_linear(self, inputs: Sequence[float]):
        vec = self._prepare_inputs(inputs)
        if self._xp is not None:
            xp = self._xp
            linear = xp.matmul(self._weights, vec) + self._bias
            return linear, vec
        outputs: List[float] = []
        for weights, bias in zip(self._weights, self._bias):
            outputs.append(sum(w * x for w, x in zip(weights, vec)) + bias)
        return outputs, vec

    # Activation -------------------------------------------------------
    def _gelu_array(self, values):
        xp = self._xp
        cubic = xp.power(values, 3)
        inner = _GELU_COEF * (values + _GELU_KAPPA * cubic)
        return 0.5 * values * (1.0 + xp.tanh(inner))

    def _gelu_derivative_array(self, values):
        xp = self._xp
        cubic = xp.power(values, 3)
        squared = xp.power(values, 2)
        inner = _GELU_COEF * (values + _GELU_KAPPA * cubic)
        tanh_inner = xp.tanh(inner)
        sech_sq = 1.0 - xp.power(tanh_inner, 2)
        term1 = 0.5 * (1.0 + tanh_inner)
        term2 = (
            0.5
            * values
            * sech_sq
            * _GELU_COEF
            * (1.0 + 3.0 * _GELU_KAPPA * squared)
        )
        return term1 + term2

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

    def _to_list(self, array) -> List[float]:
        if self._xp is not None:
            return [float(value) for value in array.tolist()]
        return [float(value) for value in array]

    # Properties --------------------------------------------------------
    @property
    def weights(self) -> List[List[float]]:
        if self._xp is not None:
            return [[float(val) for val in row] for row in self._weights.tolist()]
        return [list(row) for row in self._weights]

    @weights.setter
    def weights(self, value: Sequence[Sequence[float]]) -> None:
        if self._xp is not None:
            xp = self._xp
            self._weights = xp.asarray(value, dtype=xp.float64)
        else:
            self._weights = [list(row) for row in value]

    @property
    def bias(self) -> List[float]:
        if self._xp is not None:
            return [float(val) for val in self._bias.tolist()]
        return list(self._bias)

    @bias.setter
    def bias(self, value: Sequence[float]) -> None:
        if self._xp is not None:
            xp = self._xp
            self._bias = xp.asarray(value, dtype=xp.float64)
        else:
            self._bias = list(value)

    # Public API --------------------------------------------------------
    def predict(self, inputs: Sequence[float]) -> List[float]:
        linear, _ = self._compute_linear(inputs)
        if self._xp is not None:
            activated = self._gelu_array(linear)
            return self._to_list(activated)
        return [self._gelu_scalar(val) for val in linear]

    def predict_prob(self, inputs: Sequence[float]) -> List[float]:
        linear, _ = self._compute_linear(inputs)
        if self._xp is not None:
            xp = self._xp
            if linear.size == 0:
                return []
            activated = self._gelu_array(linear)
            max_logit = float(xp.max(activated))
            exp_vals = xp.exp(activated - max_logit)
            total = float(xp.sum(exp_vals)) or 1.0
            probs = exp_vals / total
            return self._to_list(probs)
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

        if self._xp is not None:
            xp = self._xp
            prepared_inputs = []
            for vec in paired_inputs:
                arr = self._prepare_inputs(vec)
                if not hasattr(arr, "dtype"):
                    arr = xp.asarray(list(arr), dtype=xp.float64)
                elif arr.dtype != xp.float64:
                    arr = arr.astype(xp.float64, copy=False)
                prepared_inputs.append(arr)
            inputs_mat = xp.stack(prepared_inputs, axis=0)
            targets_mat = xp.asarray(paired_targets, dtype=xp.float64)
            linear = inputs_mat.dot(self._weights.T) + self._bias
            if linear.size == 0:
                return
            activated = self._gelu_array(linear)
            max_logits = xp.max(activated, axis=1, keepdims=True)
            exp_vals = xp.exp(activated - max_logits)
            totals = xp.sum(exp_vals, axis=1, keepdims=True)
            totals = xp.maximum(totals, xp.asarray(1.0, dtype=totals.dtype))
            probs = exp_vals / totals
            errors = targets_mat - probs
            derivatives = self._gelu_derivative_array(linear)
            grad = errors * derivatives
            weight_update = grad.T.dot(inputs_mat)
            bias_update = xp.sum(grad, axis=0)
            self._weights += lr * weight_update
            self._bias += lr * bias_update
            if self.hebbian_lr:
                hebb_weights = probs.T.dot(inputs_mat)
                hebb_bias = xp.sum(probs, axis=0)
                self._weights += self.hebbian_lr * hebb_weights
                self._bias += self.hebbian_lr * hebb_bias
                self._clip_weights()
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

