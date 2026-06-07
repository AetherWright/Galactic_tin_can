"""Lightweight AI placeholders for WorldSim.

The original project shipped with an experimental mixture of neural
networks, NEAT graphs and optional Rust/C++ helpers.  For the upcoming AI
rewrite those integrations are removed in favour of tiny stubs that keep the
public API stable.  The classes below provide deterministic structures so the
simulation and unit tests continue to function while the real models are being
replaced.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Deque, Dict, Iterable, List, Optional, Sequence, Tuple
import math
import random

import yaml

from .meta_ga import RewardGA

_GELU_COEF = math.sqrt(2.0 / math.pi)
_GELU_KAPPA = 0.044715

# ---------------------------------------------------------------------------
# Compatibility flags
# ---------------------------------------------------------------------------

_USE_NEAT: bool = False
_USE_RUST: bool = False


def set_use_neat(flag: bool) -> None:
    """Retain the old API for toggling NEAT usage."""

    global _USE_NEAT
    _USE_NEAT = bool(flag)


def set_use_rust(flag: bool) -> None:
    """Retain the old API for toggling Rust helpers."""

    global _USE_RUST
    _USE_RUST = bool(flag)


# ---------------------------------------------------------------------------
# Simple perceptron controller
# ---------------------------------------------------------------------------


class SimplePerceptron:
    """Light-weight dense controller built around GeLU activations.

    This variant intentionally uses only CPU/Python primitives.  The
    GPU-accelerated implementation lives in :mod:`worldsim.cupy_ai`.
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


# ---------------------------------------------------------------------------
# Nelder-Mead layered policy
# ---------------------------------------------------------------------------


@dataclass
class _ForwardState:
    """Container used to shuttle intermediate forward pass state."""

    outputs: object
    trace: List[Tuple[List[float], List[float]]]
    hidden_outputs: List[object]



class LayeredNetwork:
    """Sparse bidirectional network with configurable hidden layers.

    Hidden layers are sparsely wired in both directions, allowing later
    layers to feed context back into earlier activations during evaluation.
    The bidirectional update runs as a lightweight refinement step layered on
    top of the original feed-forward evaluation so the public API remains
    unchanged for existing controllers.
    """

    _SPARSITY: float = 0.35
    _BIDIRECTIONAL_STEPS: int = 1

    def __init__(
        self,
        n_inputs: int,
        layer_sizes: Sequence[int],
        *,
        seed: Optional[int] = None,
    ) -> None:
        if not layer_sizes:
            raise ValueError("layer_sizes must contain at least one entry")
        self.n_inputs = n_inputs
        self.layer_sizes = [n_inputs, *layer_sizes]
        self.hidden_count = max(0, len(layer_sizes) - 1)
        self._weights: List[List[List[float]]] = []
        self._biases: List[List[float]] = []
        self._forward_masks: List[List[List[float]]] = []
        self._backward_weights: List[List[List[float]]] = []
        self._backward_masks: List[List[List[float]]] = []
        rng = random.Random(seed)
        for idx, out_size in enumerate(layer_sizes):
            in_size = self.layer_sizes[idx]
            spread = 1.0 / math.sqrt(max(1, in_size))
            mask = self._build_mask(
                out_size,
                in_size,
                rng,
                sparse=0 < idx < len(layer_sizes) - 1,
            )
            weight_rows = [
                [rng.uniform(-spread, spread) for _ in range(in_size)]
                for _ in range(out_size)
            ]
            bias_vals = [rng.uniform(-spread, spread) for _ in range(out_size)]
            self._forward_masks.append(mask)
            self._weights.append(weight_rows)
            self._biases.append(bias_vals)
            self._apply_forward_mask(idx)
        self._init_backward_connections(rng)

    # Helpers -----------------------------------------------------------
    def _build_mask(
        self,
        rows: int,
        cols: int,
        rng: random.Random,
        *,
        sparse: bool,
    ) -> List[List[float]]:
        if rows <= 0 or cols <= 0:
            return [[0.0] * cols for _ in range(rows)]
        if not sparse:
            return self._build_dense_mask(rows, cols)
        return self._build_sparse_mask(rows, cols, rng)

    def _build_sparse_mask(
        self,
        rows: int,
        cols: int,
        rng: random.Random,
    ) -> List[List[float]]:
        total = rows * cols
        target = max(rows + cols, int(total * self._SPARSITY))
        selected = set()
        for row in range(rows):
            selected.add((row, rng.randrange(cols)))
        for col in range(cols):
            selected.add((rng.randrange(rows), col))
        while len(selected) < target:
            selected.add((rng.randrange(rows), rng.randrange(cols)))
        mask = [[0.0] * cols for _ in range(rows)]
        for r, c in selected:
            mask[r][c] = 1.0
        return mask

    def _build_dense_mask(self, rows: int, cols: int) -> List[List[float]]:
        return [[1.0] * cols for _ in range(rows)]

    def _transpose_mask(self, mask: Sequence[Sequence[float]]) -> List[List[float]]:
        rows = len(mask)
        cols = len(mask[0]) if rows else 0
        return [
            [float(mask[row][col]) for row in range(rows)]
            for col in range(cols)
        ]

    def _init_backward_connections(self, rng: random.Random) -> None:
        self._backward_weights = []
        self._backward_masks = []
        if self.hidden_count <= 1:
            return
        for hid_idx in range(self.hidden_count - 1):
            mask = self._transpose_mask(self._forward_masks[hid_idx + 1])
            weights = self._initialise_backward_matrix(hid_idx, mask, rng)
            self._backward_masks.append(mask)
            self._backward_weights.append(weights)
            self._apply_backward_mask(hid_idx)

    def _initialise_backward_matrix(
        self,
        hid_idx: int,
        mask: Sequence[Sequence[float]],
        rng: random.Random,
    ) -> List[List[float]]:
        current_size = self.layer_sizes[hid_idx + 1]
        next_size = self.layer_sizes[hid_idx + 2]
        spread = 1.0 / math.sqrt(max(1, next_size))
        base = [
            [rng.uniform(-spread, spread) for _ in range(next_size)]
            for _ in range(current_size)
        ]
        # Zero-out inactive connections immediately for determinism.
        weights = [list(row) for row in base]
        for row_idx, row_mask in enumerate(mask):
            for col_idx, active in enumerate(row_mask):
                if not active:
                    weights[row_idx][col_idx] = 0.0
        return weights

    def _apply_forward_mask(self, layer_idx: int) -> None:
        if layer_idx >= len(self._forward_masks):
            return
        mask = self._forward_masks[layer_idx]
        weights = self._weights[layer_idx]
        for row_idx, row_mask in enumerate(mask):
            for col_idx, active in enumerate(row_mask):
                if not active:
                    weights[row_idx][col_idx] = 0.0

    def _apply_backward_mask(self, pair_idx: int) -> None:
        if pair_idx >= len(self._backward_masks):
            return
        mask = self._backward_masks[pair_idx]
        weights = self._backward_weights[pair_idx]
        for row_idx, row_mask in enumerate(mask):
            for col_idx, active in enumerate(row_mask):
                if not active:
                    weights[row_idx][col_idx] = 0.0

    def _prepare_inputs(self, inputs: Sequence[float]) -> List[float]:
        vec = list(inputs)[: self.n_inputs]
        if len(vec) < self.n_inputs:
            vec.extend([0.0] * (self.n_inputs - len(vec)))
        return vec

    def _to_list(self, values: Sequence[float]) -> List[float]:
        return [float(value) for value in values]

    def _compute_layers(
        self,
        inputs: Sequence[float],
        scales: Sequence[float],
        *,
        return_trace: bool,
        future_hidden: Optional[Sequence[Sequence[float]]] = None,
    ) -> _ForwardState:
        outputs = list(inputs)
        scale_values = list(scales)
        total_neurons = sum(len(layer) for layer in self._weights)
        if len(scale_values) < total_neurons:
            scale_values = [
                *scale_values,
                *([1.0] * (total_neurons - len(scale_values)))
            ]
        trace: List[Tuple[List[float], List[float]]] = []
        hidden_outputs: List[List[float]] = []
        scale_idx = 0
        for layer_idx, (weights, bias) in enumerate(zip(self._weights, self._biases)):
            layer_inputs = list(outputs)
            layer_outputs: List[float] = []
            is_output = layer_idx == len(self._weights) - 1
            for neuron_idx, neuron_weights in enumerate(weights):
                acc = bias[neuron_idx]
                for weight, value in zip(neuron_weights, layer_inputs):
                    acc += weight * value
                if (
                    future_hidden is not None
                    and layer_idx < self.hidden_count - 1
                    and layer_idx < len(self._backward_weights)
                ):
                    next_hidden = future_hidden[layer_idx + 1]
                    back_weights = self._backward_weights[layer_idx][neuron_idx]
                    back_mask = self._backward_masks[layer_idx][neuron_idx]
                    for next_idx, next_val in enumerate(next_hidden):
                        if next_idx >= len(back_weights):
                            break
                        if back_mask[next_idx]:
                            acc += back_weights[next_idx] * float(next_val)
                scale = scale_values[scale_idx] if scale_idx < len(scale_values) else 1.0
                scale_idx += 1
                acc *= scale
                layer_outputs.append(acc if is_output else math.tanh(acc))
            outputs = layer_outputs
            if layer_idx < self.hidden_count:
                hidden_outputs.append(list(outputs))
            if return_trace:
                trace.append((layer_inputs, list(layer_outputs)))
        return _ForwardState(outputs, trace, hidden_outputs)

    # Public API --------------------------------------------------------
    def forward(
        self,
        inputs: Sequence[float],
        scales: Sequence[float],
        *,
        return_trace: bool = False,
    ) -> List[float] | tuple[List[float], List[Tuple[List[float], List[float]]]]:
        vec = self._prepare_inputs(inputs)
        state = self._compute_layers(vec, scales, return_trace=return_trace)
        hidden = state.hidden_outputs
        for _ in range(self._BIDIRECTIONAL_STEPS):
            if len(hidden) < 2:
                break
            state = self._compute_layers(
                vec,
                scales,
                return_trace=return_trace,
                future_hidden=hidden,
            )
            hidden = state.hidden_outputs
        outputs = self._to_list(state.outputs)
        if return_trace:
            return outputs, state.trace
        return outputs

    def apply_hebbian(
        self,
        trace: Sequence[Tuple[Sequence[float], Sequence[float]]],
        lr: float,
        *,
        weight_clip: float = 4.0,
    ) -> None:
        if lr <= 0.0:
            return
        limit = max(1.0, weight_clip)
        for layer_idx, (inputs, outputs) in enumerate(trace):
            if layer_idx >= len(self._weights):
                break
            layer_weights = self._weights[layer_idx]
            layer_biases = self._biases[layer_idx]
            for neuron_idx, (neuron_weights, out_val) in enumerate(
                zip(layer_weights, outputs)
            ):
                for inp_idx, inp_val in enumerate(inputs):
                    if inp_idx >= len(neuron_weights):
                        break
                    neuron_weights[inp_idx] += lr * float(out_val) * float(inp_val)
                    neuron_weights[inp_idx] = max(
                        -limit, min(limit, neuron_weights[inp_idx])
                    )
                layer_biases[neuron_idx] += lr * float(out_val)
                layer_biases[neuron_idx] = max(
                    -limit, min(limit, layer_biases[neuron_idx])
                )
            self._apply_forward_mask(layer_idx)
            if (
                layer_idx < len(self._backward_weights)
                and layer_idx < self.hidden_count - 1
                and layer_idx + 1 < len(trace)
            ):
                next_outputs = trace[layer_idx + 1][1]
                back_layer = self._backward_weights[layer_idx]
                back_mask = self._backward_masks[layer_idx]
                for row_idx, out_val in enumerate(outputs):
                    if row_idx >= len(back_layer):
                        break
                    for col_idx, next_val in enumerate(next_outputs):
                        if col_idx >= len(back_layer[row_idx]):
                            break
                        if back_mask[row_idx][col_idx]:
                            back_layer[row_idx][col_idx] += lr * float(out_val) * float(next_val)
                            back_layer[row_idx][col_idx] = max(
                                -limit, min(limit, back_layer[row_idx][col_idx])
                            )
                self._apply_backward_mask(layer_idx)

    # Persistence -------------------------------------------------------
    def to_dict(self) -> Dict[str, object]:
        return {
            "layer_sizes": list(self.layer_sizes),
            "weights": self.weights,
            "biases": self.biases,
            "forward_masks": self.forward_masks,
            "backward_weights": self.backward_weights,
            "backward_masks": self.backward_masks,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "LayeredNetwork":
        layer_sizes = payload.get("layer_sizes")
        if not isinstance(layer_sizes, list) or len(layer_sizes) < 2:
            raise ValueError("invalid layer_sizes in payload")
        obj: LayeredNetwork = cls.__new__(cls)
        obj.layer_sizes = [int(v) for v in layer_sizes]
        obj.n_inputs = int(obj.layer_sizes[0])
        obj.hidden_count = max(0, len(obj.layer_sizes) - 2)
        weights = payload.get("weights")
        biases = payload.get("biases")
        if not isinstance(weights, list) or not isinstance(biases, list):
            raise ValueError("invalid network weights")
        obj._weights = [
            [[float(value) for value in row] for row in layer]
            for layer in weights
        ]
        obj._biases = [[float(value) for value in vec] for vec in biases]
        forward_masks = payload.get("forward_masks")
        if isinstance(forward_masks, list):
            obj._forward_masks = [
                [[float(value) for value in row] for row in layer]
                for layer in forward_masks
            ]
        else:
            rng = random.Random()
            obj._forward_masks = []
            total_layers = len(obj._weights)
            for idx, layer in enumerate(obj._weights):
                rows = len(layer)
                cols = len(layer[0]) if layer else 0
                sparse = 0 < idx < total_layers - 1
                obj._forward_masks.append(
                    obj._build_mask(rows, cols, rng, sparse=sparse)
                )
        for idx in range(len(obj._weights)):
            obj._apply_forward_mask(idx)
        backward_weights = payload.get("backward_weights")
        backward_masks = payload.get("backward_masks")
        if isinstance(backward_weights, list):
            obj._backward_weights = [
                [[float(value) for value in row] for row in layer]
                for layer in backward_weights
            ]
            if isinstance(backward_masks, list):
                obj._backward_masks = [
                    [[float(value) for value in row] for row in layer]
                    for layer in backward_masks
                ]
            else:
                obj._backward_masks = []
                for hid_idx in range(max(0, obj.hidden_count - 1)):
                    obj._backward_masks.append(
                        obj._transpose_mask(obj._forward_masks[hid_idx + 1])
                    )
            for idx in range(len(obj._backward_weights)):
                obj._apply_backward_mask(idx)
        else:
            obj._init_backward_connections(random.Random())
        return obj

    # Properties -------------------------------------------------------
    @property
    def weights(self) -> List[List[List[float]]]:
        return [
            [list(row) for row in layer]
            for layer in self._weights
        ]

    @weights.setter
    def weights(self, value: Sequence[Sequence[Sequence[float]]]) -> None:
        self._weights = [[list(row) for row in layer] for layer in value]
        for idx in range(len(self._weights)):
            if idx < len(self._forward_masks):
                self._apply_forward_mask(idx)

    @property
    def biases(self) -> List[List[float]]:
        return [list(layer) for layer in self._biases]

    @biases.setter
    def biases(self, value: Sequence[Sequence[float]]) -> None:
        self._biases = [list(layer) for layer in value]

    @property
    def forward_masks(self) -> List[List[List[float]]]:
        return [
            [list(row) for row in mask]
            for mask in self._forward_masks
        ]

    @forward_masks.setter
    def forward_masks(self, value: Sequence[Sequence[Sequence[float]]]) -> None:
        self._forward_masks = [[list(row) for row in layer] for layer in value]
        for idx in range(len(self._weights)):
            self._apply_forward_mask(idx)

    @property
    def backward_weights(self) -> List[List[List[float]]]:
        return [
            [list(row) for row in layer]
            for layer in self._backward_weights
        ]

    @backward_weights.setter
    def backward_weights(self, value: Sequence[Sequence[Sequence[float]]]) -> None:
        self._backward_weights = [
            [list(row) for row in layer]
            for layer in value
        ]
        for idx in range(len(self._backward_weights)):
            self._apply_backward_mask(idx)

    @property
    def backward_masks(self) -> List[List[List[float]]]:
        return [
            [list(row) for row in mask]
            for mask in self._backward_masks
        ]

    @backward_masks.setter
    def backward_masks(self, value: Sequence[Sequence[Sequence[float]]]) -> None:
        self._backward_masks = [
            [list(row) for row in layer]
            for layer in value
        ]
        for idx in range(len(self._backward_weights)):
            self._apply_backward_mask(idx)
@dataclass
class SimplexResult:
    point: List[float]
    value: float
    iterations: int
    evaluations: int
    converged: bool


def nelder_mead_minimise(
    func: Callable[[Sequence[float]], float],
    initial: Sequence[float],
    *,
    step: float = 0.1,
    max_iter: int = 50,
    tol: float = 1e-3,
) -> SimplexResult:
    dim = len(initial)
    if dim == 0:
        raise ValueError("initial point must have at least one dimension")
    simplex: List[List[float]] = [list(initial)]
    for idx in range(dim):
        vertex = list(initial)
        vertex[idx] += step if step else 0.1
        simplex.append(vertex)
    values = [func(vertex) for vertex in simplex]
    evaluations = len(values)
    iterations = 0
    rho = 1.0
    chi = 2.0
    gamma = 0.5
    sigma = 0.5
    converged = False

    while iterations < max_iter:
        order = sorted(range(len(simplex)), key=lambda idx: values[idx])
        simplex = [simplex[idx] for idx in order]
        values = [values[idx] for idx in order]
        best_value = values[0]
        worst_value = values[-1]
        spread = max(abs(v - best_value) for v in values[1:]) if len(values) > 1 else 0.0
        if spread < tol:
            converged = True
            break

        centroid = [
            sum(simplex[i][d] for i in range(len(simplex) - 1)) / dim
            for d in range(dim)
        ]
        worst = simplex[-1]

        reflection = [centroid[d] + rho * (centroid[d] - worst[d]) for d in range(dim)]
        reflection_value = func(reflection)
        evaluations += 1

        if reflection_value < values[0]:
            expansion = [
                centroid[d] + chi * (reflection[d] - centroid[d]) for d in range(dim)
            ]
            expansion_value = func(expansion)
            evaluations += 1
            if expansion_value < reflection_value:
                simplex[-1] = expansion
                values[-1] = expansion_value
            else:
                simplex[-1] = reflection
                values[-1] = reflection_value
        elif reflection_value < values[-2]:
            simplex[-1] = reflection
            values[-1] = reflection_value
        else:
            if reflection_value < worst_value:
                contraction = [
                    centroid[d] + gamma * (reflection[d] - centroid[d])
                    for d in range(dim)
                ]
            else:
                contraction = [
                    centroid[d] + gamma * (worst[d] - centroid[d]) for d in range(dim)
                ]
            contraction_value = func(contraction)
            evaluations += 1
            if contraction_value < worst_value:
                simplex[-1] = contraction
                values[-1] = contraction_value
            else:
                best = simplex[0]
                for i in range(1, len(simplex)):
                    simplex[i] = [
                        best[d] + sigma * (simplex[i][d] - best[d]) for d in range(dim)
                    ]
                    values[i] = func(simplex[i])
                evaluations += len(simplex) - 1

        iterations += 1

    result = SimplexResult(
        point=list(simplex[0]),
        value=float(values[0]),
        iterations=iterations,
        evaluations=evaluations,
        converged=converged,
    )
    return result


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


# ---------------------------------------------------------------------------
# Role-specific facades
# ---------------------------------------------------------------------------


class WarAI(NelderMeadPolicy):
    """Layered Nelder-Mead controller for the military AI."""

    def __init__(
        self,
        *,
        table_path: str | Path | None = None,
        grid_size: int = 8,
        allies_dim: int = 4,
        epsilon: float = 0.1,
        gamma: float = 0.95,
        hidden_layers: Sequence[int] = (20, 16, 12, 9, 6, 4, 6, 9, 12, 16, 20),
        memory_size: int = 32,
    ) -> None:
        self.grid_size = max(1, grid_size)
        self.allies_dim = max(1, allies_dim)
        self.grid_feature_count = self.grid_size * self.grid_size * 2
        n_inputs = 4 + self.allies_dim + self.grid_feature_count
        super().__init__(
            n_inputs,
            2,
            table_path=table_path,
            epsilon=epsilon,
            gamma=gamma,
            hidden_layers=hidden_layers,
            memory_size=memory_size,
        )

    def set_allies_dimension(self, allies_dim: int) -> None:
        """Adjust the expected size of the ally feature vector."""

        allies_dim = max(1, allies_dim)
        if allies_dim == self.allies_dim:
            return
        self.allies_dim = allies_dim
        self.n_inputs = 4 + self.allies_dim + self.grid_feature_count
        self.network = LayeredNetwork(
            self.n_inputs,
            [*self.hidden_layers, self.n_actions],
            seed=self._rng.randint(0, 2**31 - 1),
        )
        self.scale_count = self._expected_scale_length()
        self.scales = [1.0] * self.scale_count
        self.memory.clear()

    def create_doctrine(self) -> str:
        styles = ["Blitz", "Trench", "Guerilla", "Mobile", "Integrated"]
        focuses = ["Assault", "Defense", "Strategy", "Doctrine"]
        return f"{random.choice(styles)} {random.choice(focuses)}"


class DomesticPolicyAI(NelderMeadPolicy):
    """Nelder-Mead based learner for domestic policy."""

    def __init__(
        self,
        actions: int = 4,
        n_inputs: int = 5,
        *,
        table_path: str | Path | None = None,
        epsilon: float = 0.1,
        gamma: float = 0.95,
        hidden_layers: Sequence[int] = (12, 9, 7, 5, 4, 5, 7, 9, 12),
    ) -> None:
        super().__init__(
            n_inputs,
            actions,
            table_path=table_path,
            epsilon=epsilon,
            gamma=gamma,
            hidden_layers=hidden_layers,
        )


class CivilianOverseerAI(NelderMeadPolicy):
    """Nelder-Mead overseer: routes to one of the N civilian departments."""

    def __init__(
        self,
        n_depts: int = 6,
        n_inputs: int = 22,
        *,
        table_path: str | Path | None = None,
        epsilon: float = 0.10,
        gamma: float = 0.95,
        hidden_layers: Sequence[int] = (16, 12, 8, 6, 8, 12, 16),
    ) -> None:
        super().__init__(
            n_inputs, n_depts,
            table_path=table_path, epsilon=epsilon, gamma=gamma,
            hidden_layers=hidden_layers,
        )


class DepartmentPolicyAI(NelderMeadPolicy):
    """Nelder-Mead learner for a single department's local action sub-space."""

    def __init__(
        self,
        slug: str,
        n_actions: int,
        n_inputs: int = 22,
        *,
        table_path: str | Path | None = None,
        epsilon: float = 0.10,
        gamma: float = 0.95,
        hidden_layers: Sequence[int] = (12, 9, 7, 5, 7, 9, 12),
    ) -> None:
        super().__init__(
            n_inputs, n_actions,
            table_path=table_path, epsilon=epsilon, gamma=gamma,
            hidden_layers=hidden_layers,
        )
        self.slug = slug


class ProjectAI(NelderMeadPolicy):
    """Nelder-Mead learner for construction project decisions."""

    def __init__(
        self,
        actions: int,
        n_inputs: int = 6,
        *,
        table_path: str | Path | None = None,
        epsilon: float = 0.1,
        gamma: float = 0.95,
        hidden_layers: Sequence[int] = (14, 11, 9, 7, 5, 4, 5, 7 ,9, 11, 14),
    ) -> None:
        super().__init__(
            n_inputs,
            actions,
            table_path=table_path,
            epsilon=epsilon,
            gamma=gamma,
            hidden_layers=hidden_layers,
        )


class DiplomacyAI(NelderMeadPolicy):
    """Nelder-Mead learner for diplomacy choices."""

    def __init__(
        self,
        actions: int = 3,
        n_inputs: int = 5,
        *,
        table_path: str | Path | None = None,
        epsilon: float = 0.1,
        gamma: float = 0.9,
        hidden_layers: Sequence[int] = (10, 8, 6, 4, 3, 4, 6, 8, 10),
    ) -> None:
        super().__init__(
            n_inputs,
            actions,
            table_path=table_path,
            epsilon=epsilon,
            gamma=gamma,
            hidden_layers=hidden_layers,
        )


class ResearchAI(NelderMeadPolicy):
    """Nelder-Mead learner for technology selection."""

    def __init__(
        self,
        actions: int,
        n_inputs: int = 4,
        *,
        table_path: str | Path | None = None,
        epsilon: float = 0.1,
        gamma: float = 0.9,
        hidden_layers: Sequence[int] = (10, 8, 6, 4, 3, 4, 6, 8, 10),
    ) -> None:
        super().__init__(
            n_inputs,
            actions,
            table_path=table_path,
            epsilon=epsilon,
            gamma=gamma,
            hidden_layers=hidden_layers,
        )


# ---------------------------------------------------------------------------
# Strategic planner using Nelder-Mead
# ---------------------------------------------------------------------------


MetricVector = Tuple[float, float, float, float, float]
MetricFn = Callable[[Sequence[float]], MetricVector]
PenaltyFn = Callable[[Sequence[float], MetricVector], float]


@dataclass
class NelderMeadResult:
    """Container describing the outcome of a Nelder-Mead optimisation run."""

    optimum: List[float]
    loss: float
    iterations: int
    evaluations: int
    converged: bool


class StrategicNelderMeadAI:
    """Continuous planner optimised against meta-learned modifiers.

    The planner minimises the loss function::

        L = -(α * log(1 + E) + β * log(1 + M) + γ * log(1 + T)
              + δ * S + ϵ * C) + penalties

    ``α``..``ϵ`` are supplied by a :class:`~worldsim.meta_ga.RewardGA`
    instance which allows the meta-genetic algorithm to tune behaviour over
    long simulations.  ``E``..``C`` are economy, military, technology,
    stability and culture metrics produced by a user supplied evaluator.
    ``penalties`` represent soft constraints that can steer the search toward
    valid solutions (for example budget limits or safety checks).
    """

    def __init__(
        self,
        meta_ga: RewardGA,
        *,
        step_scale: float = 0.1,
        tol: float = 1e-4,
        max_iter: int = 200,
        metric_fn: Optional[MetricFn] = None,
        penalty_fn: Optional[PenaltyFn] = None,
    ) -> None:
        self.meta_ga = meta_ga
        self.step_scale = max(1e-6, step_scale)
        self.tol = max(1e-8, tol)
        self.max_iter = max(1, max_iter)
        self.metric_fn = metric_fn or self._default_metric_fn
        self.penalty_fn = penalty_fn or self._default_penalty
        self.last_result: Optional[NelderMeadResult] = None

    # Helpers ------------------------------------------------------------
    def _default_metric_fn(self, vector: Sequence[float]) -> MetricVector:
        data = list(vector[:5])
        if len(data) < 5:
            data.extend([0.0] * (5 - len(data)))
        return (
            float(data[0]),
            float(data[1]),
            float(data[2]),
            float(data[3]),
            float(data[4]),
        )

    def _default_penalty(
        self, vector: Sequence[float], metrics: MetricVector
    ) -> float:
        penalty = 0.0
        # Discourage negative outcome projections.
        for value in metrics:
            if value < 0.0:
                penalty += abs(value) * 5.0
        # Keep decision vectors from exploding; useful for soft budgets.
        total = sum(vector)
        if total > 1.0:
            penalty += (total - 1.0) * 2.0
        return penalty

    def _meta_modifiers(self) -> MetricVector:
        weights = list(self.meta_ga.weights)
        if len(weights) < 5:
            weights.extend([1.0] * (5 - len(weights)))
        modifiers = [max(0.01, abs(w)) for w in weights[:5]]
        return (
            float(modifiers[0]),
            float(modifiers[1]),
            float(modifiers[2]),
            float(modifiers[3]),
            float(modifiers[4]),
        )

    def _loss(
        self,
        vector: Sequence[float],
        metric_fn: MetricFn,
        penalty_fn: PenaltyFn,
    ) -> float:
        metrics = metric_fn(vector)
        alpha, beta, gamma, delta, epsilon = self._meta_modifiers()
        econ, mil, tech, stab, cult = metrics
        econ = max(-0.999999, econ)
        mil = max(-0.999999, mil)
        tech = max(-0.999999, tech)
        base = (
            alpha * math.log1p(econ)
            + beta * math.log1p(mil)
            + gamma * math.log1p(tech)
            + delta * stab
            + epsilon * cult
        )
        penalty = penalty_fn(vector, metrics)
        return -base + penalty

    def _build_simplex(self, guess: Sequence[float]) -> List[List[float]]:
        base = list(guess)
        dim = len(base)
        if dim == 0:
            raise ValueError("initial guess must contain at least one dimension")
        simplex: List[List[float]] = [base]
        for idx in range(dim):
            vertex = list(base)
            step = self.step_scale * (abs(base[idx]) + 1.0)
            vertex[idx] = vertex[idx] + step
            simplex.append(vertex)
        return simplex

    # Public API --------------------------------------------------------
    def evaluate(self, vector: Sequence[float]) -> float:
        """Return the loss for ``vector`` using stored callbacks."""

        return self._loss(vector, self.metric_fn, self.penalty_fn)

    def plan(
        self,
        initial_guess: Sequence[float],
        *,
        metric_fn: Optional[MetricFn] = None,
        penalty_fn: Optional[PenaltyFn] = None,
        max_iter: Optional[int] = None,
    ) -> NelderMeadResult:
        """Optimise the decision vector starting from ``initial_guess``."""

        metric_cb = metric_fn or self.metric_fn
        penalty_cb = penalty_fn or self.penalty_fn
        simplex = self._build_simplex(initial_guess)
        values = [self._loss(v, metric_cb, penalty_cb) for v in simplex]
        evaluations = len(values)
        iterations = 0
        limit = max_iter if max_iter is not None else self.max_iter
        rho = 1.0  # reflection
        chi = 2.0  # expansion
        gamma = 0.5  # contraction
        sigma = 0.5  # shrink
        converged = False

        while iterations < limit:
            order = sorted(range(len(simplex)), key=lambda idx: values[idx])
            simplex = [simplex[idx] for idx in order]
            values = [values[idx] for idx in order]
            best_value = values[0]
            worst_value = values[-1]
            # Convergence check based on simplex value spread.
            if max(abs(v - best_value) for v in values[1:]) < self.tol:
                converged = True
                break

            dim = len(simplex[0])
            centroid = [
                sum(simplex[i][d] for i in range(len(simplex) - 1)) / dim
                for d in range(dim)
            ]
            worst = simplex[-1]

            reflection = [
                centroid[d] + rho * (centroid[d] - worst[d]) for d in range(dim)
            ]
            reflection_value = self._loss(reflection, metric_cb, penalty_cb)
            evaluations += 1

            if reflection_value < values[0]:
                expansion = [
                    centroid[d] + chi * (reflection[d] - centroid[d])
                    for d in range(dim)
                ]
                expansion_value = self._loss(expansion, metric_cb, penalty_cb)
                evaluations += 1
                if expansion_value < reflection_value:
                    simplex[-1] = expansion
                    values[-1] = expansion_value
                else:
                    simplex[-1] = reflection
                    values[-1] = reflection_value
            elif reflection_value < values[-2]:
                simplex[-1] = reflection
                values[-1] = reflection_value
            else:
                if reflection_value < worst_value:
                    contraction = [
                        centroid[d] + gamma * (reflection[d] - centroid[d])
                        for d in range(dim)
                    ]
                    contraction_value = self._loss(contraction, metric_cb, penalty_cb)
                else:
                    contraction = [
                        centroid[d] + gamma * (worst[d] - centroid[d])
                        for d in range(dim)
                    ]
                    contraction_value = self._loss(contraction, metric_cb, penalty_cb)
                evaluations += 1
                if contraction_value < worst_value:
                    simplex[-1] = contraction
                    values[-1] = contraction_value
                else:
                    best = simplex[0]
                    for i in range(1, len(simplex)):
                        simplex[i] = [
                            best[d] + sigma * (simplex[i][d] - best[d])
                            for d in range(dim)
                        ]
                        values[i] = self._loss(simplex[i], metric_cb, penalty_cb)
                    evaluations += len(simplex) - 1

            iterations += 1

        optimum = simplex[0]
        loss = values[0]
        result = NelderMeadResult(
            optimum=list(optimum),
            loss=float(loss),
            iterations=iterations,
            evaluations=evaluations,
            converged=converged,
        )
        self.last_result = result
        return result

    def interpret(self, vector: Sequence[float]) -> Dict[str, float]:
        """Return diagnostic breakdown of the loss contributions."""

        metrics = self.metric_fn(vector)
        alpha, beta, gamma, delta, epsilon = self._meta_modifiers()
        econ, mil, tech, stab, cult = metrics
        econ = max(-0.999999, econ)
        mil = max(-0.999999, mil)
        tech = max(-0.999999, tech)
        penalty = self.penalty_fn(vector, metrics)
        return {
            "economy": alpha * math.log1p(econ),
            "military": beta * math.log1p(mil),
            "technology": gamma * math.log1p(tech),
            "stability": delta * stab,
            "culture": epsilon * cult,
            "penalty": penalty,
            "loss": -(
                alpha * math.log1p(econ)
                + beta * math.log1p(mil)
                + gamma * math.log1p(tech)
                + delta * stab
                + epsilon * cult
            )
            + penalty,
        }
