"""Per-century nation fitness scoring.

A standalone diagnostic metric: rewards stability, expansion and a balanced
military without runaway militarisation. Not read by the RL training loop
(see ``Nation.compute_reward`` / ``Nation.reward_snapshot`` for that) — kept
here for reporting/analysis.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from ..nations import Nation


def century_score(n: "Nation") -> int:
    """Return a per-century fitness score summarising *n*'s overall standing."""
    score = 0
    # Economy health
    if n.economy < 10.0:
        score -= 2
    if n.economy > 10.0:
        score += 2
    # Stability
    if n.stability > 75.0:
        score += 20
    if n.stability > 90.0:
        score += 20
    if n.stability > 50.0:
        score += 20
    if n.stability < 50.0:
        score -= 20
    if n.stability < 40.0:
        score -= 20
    if n.stability < 30.0:
        score -= 20
    if n.stability < 20.0:
        score -= 20
    if n.stability < 10.0:
        score -= 20
    # Star expansion
    score += 3 * n.star_count
    # Fleet presence
    score += 4 * len(n.fleets)
    # Military balance - has military but not at expense of economy
    if n.military > 5.0 and n.economy > 5.0:
        score += 1
    if n.military < 5.0 and n.economy > 5.0:
        score -= 10
        score -= 4 * len(n.divisions)
    return score
