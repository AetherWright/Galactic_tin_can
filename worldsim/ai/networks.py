"""Sparse bidirectional layered network used by the Nelder-Mead policies.

Runs vectorised on the active array backend (CuPy on GPU, otherwise NumPy)
and falls back to pure-Python lists when neither is available.  See
:mod:`worldsim.ai._backend` for the ``WORLDSIM_AI_BACKEND`` override.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import math
import random

from ._backend import ai_array_module


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
        self._xp = ai_array_module()
        self._weights: List[object] = []
        self._biases: List[object] = []
        self._forward_masks: List[object] = []
        self._backward_weights: List[object] = []
        self._backward_masks: List[object] = []
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
            if self._xp is not None:
                xp = self._xp
                mask_arr = xp.asarray(mask, dtype=xp.float64)
                weights_arr = xp.asarray(weight_rows, dtype=xp.float64)
                bias_arr = xp.asarray(bias_vals, dtype=xp.float64)
                self._forward_masks.append(mask_arr)
                self._weights.append(weights_arr)
                self._biases.append(bias_arr)
            else:
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

    def _transpose_mask(self, mask: object) -> object:
        if self._xp is not None:
            xp = self._xp
            return xp.asarray(mask, dtype=xp.float64).T.copy()
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
        mask: object,
        rng: random.Random,
    ) -> object:
        current_size = self.layer_sizes[hid_idx + 1]
        next_size = self.layer_sizes[hid_idx + 2]
        spread = 1.0 / math.sqrt(max(1, next_size))
        base = [
            [rng.uniform(-spread, spread) for _ in range(next_size)]
            for _ in range(current_size)
        ]
        if self._xp is not None:
            xp = self._xp
            return xp.asarray(base, dtype=xp.float64)
        return base

    def _apply_forward_mask(self, layer_idx: int) -> None:
        if layer_idx >= len(self._forward_masks):
            return
        mask = self._forward_masks[layer_idx]
        weights = self._weights[layer_idx]
        if self._xp is not None:
            xp = self._xp
            xp.multiply(weights, mask, out=weights)
            return
        for row_idx, row_mask in enumerate(mask):
            for col_idx, active in enumerate(row_mask):
                if not active:
                    weights[row_idx][col_idx] = 0.0

    def _apply_backward_mask(self, pair_idx: int) -> None:
        if pair_idx >= len(self._backward_masks):
            return
        mask = self._backward_masks[pair_idx]
        weights = self._backward_weights[pair_idx]
        if self._xp is not None:
            xp = self._xp
            xp.multiply(weights, mask, out=weights)
            return
        for row_idx, row_mask in enumerate(mask):
            for col_idx, active in enumerate(row_mask):
                if not active:
                    weights[row_idx][col_idx] = 0.0

    def _prepare_inputs(self, inputs: Sequence[float]) -> List[float]:
        vec = list(inputs)[: self.n_inputs]
        if len(vec) < self.n_inputs:
            vec.extend([0.0] * (self.n_inputs - len(vec)))
        if self._xp is not None:
            xp = self._xp
            return xp.asarray(vec, dtype=xp.float64)
        return vec

    def _to_list(self, values: object) -> List[float]:
        if self._xp is not None:
            xp = self._xp
            arr = values
            if not hasattr(arr, "tolist"):
                arr = xp.asarray(list(values), dtype=xp.float64)
            data = arr.tolist()
            if isinstance(data, list):
                return [float(val) for val in data]
            return [float(data)]
        return [float(val) for val in values]

    def _compute_layers(
        self,
        inputs: object,
        scales: Sequence[float],
        *,
        return_trace: bool,
        future_hidden: Optional[Sequence[object]] = None,
    ) -> _ForwardState:
        if self._xp is not None:
            xp = self._xp
            outputs = inputs
            total_neurons = sum(layer.shape[0] for layer in self._weights)
            if total_neurons:
                scales_array = xp.ones(total_neurons, dtype=xp.float64)
                if scales:
                    provided = xp.asarray(list(scales), dtype=xp.float64)
                    limit = min(int(provided.size), total_neurons)
                    if limit:
                        scales_array[:limit] = provided[:limit]
            else:
                scales_array = xp.zeros(0, dtype=xp.float64)
            scale_idx = 0
            trace: List[Tuple[List[float], List[float]]] = []
            hidden_outputs: List[object] = []
            for layer_idx, (weights, bias) in enumerate(zip(self._weights, self._biases)):
                layer_inputs = outputs
                linear = weights.dot(outputs) + bias
                if (
                    future_hidden is not None
                    and layer_idx < self.hidden_count - 1
                    and layer_idx < len(self._backward_weights)
                ):
                    next_hidden = future_hidden[layer_idx + 1]
                    if not hasattr(next_hidden, "dtype"):
                        next_hidden = xp.asarray(list(next_hidden), dtype=xp.float64)
                    backward = self._backward_weights[layer_idx].dot(next_hidden)
                    linear = linear + backward
                if linear.size:
                    layer_scale = scales_array[scale_idx : scale_idx + linear.size]
                    if layer_scale.size < linear.size:
                        pad = xp.ones(linear.size - layer_scale.size, dtype=xp.float64)
                        layer_scale = xp.concatenate([layer_scale, pad])
                    linear = linear * layer_scale
                scale_idx += linear.size
                is_output = layer_idx == len(self._weights) - 1
                outputs = linear if is_output else xp.tanh(linear)
                if layer_idx < self.hidden_count:
                    hidden_outputs.append(outputs.copy())
                if return_trace:
                    trace.append((self._to_list(layer_inputs), self._to_list(outputs)))
            return _ForwardState(outputs, trace, hidden_outputs)

        outputs = list(inputs)
        scale_values = list(scales)
        total_neurons = sum(len(layer) for layer in self._weights)
        if len(scale_values) < total_neurons:
            scale_values = [*scale_values, *([1.0] * (total_neurons - len(scale_values)))]
        trace = []
        hidden_outputs = []
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
        if self._xp is not None:
            xp = self._xp
            for layer_idx, (inputs, outputs) in enumerate(trace):
                if layer_idx >= len(self._weights):
                    break
                weights = self._weights[layer_idx]
                biases = self._biases[layer_idx]
                inp = xp.asarray(list(inputs), dtype=xp.float64)
                out = xp.asarray(list(outputs), dtype=xp.float64)
                if inp.size < weights.shape[1]:
                    pad = xp.zeros(weights.shape[1] - int(inp.size), dtype=xp.float64)
                    inp = xp.concatenate([inp, pad])
                if inp.size > weights.shape[1]:
                    inp = inp[: weights.shape[1]]
                if out.size < weights.shape[0]:
                    pad = xp.zeros(weights.shape[0] - int(out.size), dtype=xp.float64)
                    out = xp.concatenate([out, pad])
                if out.size > weights.shape[0]:
                    out = out[: weights.shape[0]]
                weights += lr * xp.outer(out, inp)
                self._apply_forward_mask(layer_idx)
                biases += lr * out
                xp.clip(weights, -limit, limit, out=weights)
                xp.clip(biases, -limit, limit, out=biases)
                if (
                    layer_idx < len(self._backward_weights)
                    and layer_idx < self.hidden_count - 1
                    and layer_idx + 1 < len(trace)
                ):
                    next_outputs = xp.asarray(
                        list(trace[layer_idx + 1][1]), dtype=xp.float64
                    )
                    back_weights = self._backward_weights[layer_idx]
                    if next_outputs.size < back_weights.shape[1]:
                        pad = xp.zeros(
                            back_weights.shape[1] - int(next_outputs.size),
                            dtype=xp.float64,
                        )
                        next_outputs = xp.concatenate([next_outputs, pad])
                    if next_outputs.size > back_weights.shape[1]:
                        next_outputs = next_outputs[: back_weights.shape[1]]
                    current = out[: back_weights.shape[0]]
                    back_weights += lr * xp.outer(current, next_outputs)
                    self._apply_backward_mask(layer_idx)
                    xp.clip(back_weights, -limit, limit, out=back_weights)
            return

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
        xp = ai_array_module()
        parsed_weights = [
            [[float(value) for value in row] for row in layer]
            for layer in weights
        ]
        parsed_biases = [[float(value) for value in vec] for vec in biases]
        obj._xp = xp
        if xp is not None:
            obj._weights = [xp.asarray(layer, dtype=xp.float64) for layer in parsed_weights]
            obj._biases = [xp.asarray(vec, dtype=xp.float64) for vec in parsed_biases]
        else:
            obj._weights = parsed_weights
            obj._biases = parsed_biases
        forward_masks = payload.get("forward_masks")
        if isinstance(forward_masks, list):
            parsed_masks = [
                [[float(value) for value in row] for row in layer]
                for layer in forward_masks
            ]
        else:
            rng = random.Random()
            parsed_masks = []
            total_layers = len(parsed_weights)
            for idx, layer in enumerate(parsed_weights):
                rows = len(layer)
                cols = len(layer[0]) if layer else 0
                sparse = 0 < idx < total_layers - 1
                parsed_masks.append(obj._build_mask(rows, cols, rng, sparse=sparse))
        if xp is not None:
            obj._forward_masks = [xp.asarray(layer, dtype=xp.float64) for layer in parsed_masks]
        else:
            obj._forward_masks = parsed_masks
        for idx in range(len(obj._weights)):
            obj._apply_forward_mask(idx)
        backward_weights = payload.get("backward_weights")
        backward_masks = payload.get("backward_masks")
        obj._backward_weights = []
        obj._backward_masks = []
        if isinstance(backward_weights, list):
            parsed_back_weights = [
                [[float(value) for value in row] for row in layer]
                for layer in backward_weights
            ]
            if xp is not None:
                obj._backward_weights = [
                    xp.asarray(layer, dtype=xp.float64) for layer in parsed_back_weights
                ]
            else:
                obj._backward_weights = parsed_back_weights
            if isinstance(backward_masks, list):
                parsed_back_masks = [
                    [[float(value) for value in row] for row in layer]
                    for layer in backward_masks
                ]
                if xp is not None:
                    obj._backward_masks = [
                        xp.asarray(layer, dtype=xp.float64) for layer in parsed_back_masks
                    ]
                else:
                    obj._backward_masks = parsed_back_masks
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
        if self._xp is not None:
            return [
                [[float(value) for value in row] for row in layer.tolist()]
                for layer in self._weights
            ]
        return [
            [list(row) for row in layer]
            for layer in self._weights
        ]

    @weights.setter
    def weights(self, value: Sequence[Sequence[Sequence[float]]]) -> None:
        if self._xp is not None:
            xp = self._xp
            self._weights = [xp.asarray(layer, dtype=xp.float64) for layer in value]
        else:
            self._weights = [[list(row) for row in layer] for layer in value]
        for idx in range(len(self._weights)):
            if idx < len(self._forward_masks):
                self._apply_forward_mask(idx)

    @property
    def biases(self) -> List[List[float]]:
        if self._xp is not None:
            return [[float(v) for v in layer.tolist()] for layer in self._biases]
        return [list(layer) for layer in self._biases]

    @biases.setter
    def biases(self, value: Sequence[Sequence[float]]) -> None:
        if self._xp is not None:
            xp = self._xp
            self._biases = [xp.asarray(layer, dtype=xp.float64) for layer in value]
        else:
            self._biases = [list(layer) for layer in value]

    @property
    def forward_masks(self) -> List[List[List[float]]]:
        if self._xp is not None:
            return [
                [[float(value) for value in row] for row in mask.tolist()]
                for mask in self._forward_masks
            ]
        return [
            [list(row) for row in mask]
            for mask in self._forward_masks
        ]

    @forward_masks.setter
    def forward_masks(self, value: Sequence[Sequence[Sequence[float]]]) -> None:
        if self._xp is not None:
            xp = self._xp
            self._forward_masks = [xp.asarray(layer, dtype=xp.float64) for layer in value]
        else:
            self._forward_masks = [[list(row) for row in layer] for layer in value]
        for idx in range(len(self._weights)):
            self._apply_forward_mask(idx)

    @property
    def backward_weights(self) -> List[List[List[float]]]:
        if self._xp is not None:
            return [
                [[float(value) for value in row] for row in layer.tolist()]
                for layer in self._backward_weights
            ]
        return [
            [list(row) for row in layer]
            for layer in self._backward_weights
        ]

    @backward_weights.setter
    def backward_weights(self, value: Sequence[Sequence[Sequence[float]]]) -> None:
        if self._xp is not None:
            xp = self._xp
            self._backward_weights = [
                xp.asarray(layer, dtype=xp.float64) for layer in value
            ]
        else:
            self._backward_weights = [[list(row) for row in layer] for layer in value]
        for idx in range(len(self._backward_weights)):
            self._apply_backward_mask(idx)

    @property
    def backward_masks(self) -> List[List[List[float]]]:
        if self._xp is not None:
            return [
                [[float(value) for value in row] for row in mask.tolist()]
                for mask in self._backward_masks
            ]
        return [
            [list(row) for row in mask]
            for mask in self._backward_masks
        ]

    @backward_masks.setter
    def backward_masks(self, value: Sequence[Sequence[Sequence[float]]]) -> None:
        if self._xp is not None:
            xp = self._xp
            self._backward_masks = [
                xp.asarray(layer, dtype=xp.float64) for layer in value
            ]
        else:
            self._backward_masks = [[list(row) for row in layer] for layer in value]
        for idx in range(len(self._backward_weights)):
            self._apply_backward_mask(idx)
