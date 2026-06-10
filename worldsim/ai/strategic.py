"""Continuous strategic planner optimised against meta-learned modifiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import math

from .meta_ga import RewardGA


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
