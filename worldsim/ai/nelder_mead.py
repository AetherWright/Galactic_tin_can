"""Plain Nelder-Mead simplex minimiser."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence


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

