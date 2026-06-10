"""Contextual Q-table for event response learning."""

from __future__ import annotations

import random
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# EventQLearner — contextual action-value table
# ---------------------------------------------------------------------------

class EventQLearner:
    """Contextual Q-table for event response learning.

    State context is bucketed into 18 bins to keep the tables small
    while still capturing economy, stability, and war status:

    * ``economy_level``  – 0 (< 200), 1 (< 600), 2 (≥ 600)
    * ``stability_level`` – 0 (< 40),  1 (< 70),  2 (≥ 70)
    * ``at_war``         – 0 / 1

    Each (event, state_key) → list of Q-values, one per choice.
    Updates use simple TD(0):

        Q[choice] += ALPHA * (reward - Q[choice])
    """

    ALPHA:   float = 0.25    # learning rate — fast enough for short runs
    EPSILON: float = 0.15    # exploration rate

    def __init__(self) -> None:
        # tables[event][state_key] = [q_val_per_choice]
        self.tables: Dict[str, Dict[str, List[float]]] = {}

    # ------------------------------------------------------------------
    # State bucketing
    # ------------------------------------------------------------------

    @staticmethod
    def _state_key(nation) -> str:
        econ = (0 if nation.economy_linear < 200
                else (1 if nation.economy_linear < 600 else 2))
        stab = (0 if nation.stability < 40
                else (1 if nation.stability < 70 else 2))
        war  = 1 if nation.at_war else 0
        return f"{econ},{stab},{war}"

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def choose(self, nation, event: str, n_choices: int) -> Optional[int]:
        """Return the best action index, or ``None`` if the table is empty.

        Falls back to the caller’s heuristic when no data exists for the
        current (event, state) pair.
        """
        if n_choices <= 0:
            return None
        key   = self._state_key(nation)
        q_map = self.tables.get(event)
        if not q_map:
            return None
        q_vals = q_map.get(key)
        if not q_vals:
            return None
        if random.random() < self.EPSILON:
            return random.randrange(n_choices)
        # Pad to n_choices in case the table was built with a different count
        q = (list(q_vals) + [0.0] * n_choices)[:n_choices]
        return q.index(max(q))

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def update(
        self,
        nation,
        event:    str,
        choice:   int,
        reward:   float,
        n_choices: int,
    ) -> None:
        """TD(0) update: Q[choice] += ALPHA * (reward - Q[choice])."""
        if n_choices <= 0:
            return
        key   = self._state_key(nation)
        table = self.tables.setdefault(event, {})
        q_vals = table.get(key)
        if q_vals is None:
            q_vals = [0.0] * n_choices
            table[key] = q_vals
        # Grow to accommodate choice index if needed
        while len(q_vals) < n_choices:
            q_vals.append(0.0)
        idx = max(0, min(choice, n_choices - 1))
        q_vals[idx] += self.ALPHA * (reward - q_vals[idx])

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Dict[str, List[float]]]:
        return {
            event: dict(entries)
            for event, entries in self.tables.items()
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "EventQLearner":
        obj = cls()
        for event, entries in data.items():
            if not isinstance(entries, dict):
                continue
            obj.tables[event] = {
                key: [float(v) for v in vals]
                for key, vals in entries.items()
                if isinstance(vals, list)
            }
        return obj

