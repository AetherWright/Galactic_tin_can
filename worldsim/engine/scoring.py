"""Per-century MetaGA fitness scoring.

The score feeds every nation's RewardGA genomes; it rewards stability,
expansion and a balanced military without runaway militarisation.
"""

from __future__ import annotations

from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from ..nations import Nation


def century_score(n: "Nation") -> int:
    """Return the MetaGA fitness contribution for one century."""
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


def apply_century_scores(nations: Dict[int, "Nation"]) -> None:
    """Step every nation's RewardGA instances with this century's score."""
    for n in nations.values():
        score = century_score(n)
        for ga in n.reward_ga.values():
            ga.step(score)
