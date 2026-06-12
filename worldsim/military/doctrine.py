"""Strategic doctrine: the DoctrineSignal vocabulary and DoctrineAI.

DoctrineAI runs at the **end** of each nation's ``process_turn``, *after* the
civilian and diplomacy AIs have already executed.  It therefore sees the
civilian AI's latest action choice (``nation.last_civilian_action``) and the
current alliance / war state shaped by diplomacy.  Its output is a
:class:`DoctrineSignal` stored on ``nation.doctrine_signal``.

Division doctrine integration
    ``issue_orders`` in :mod:`worldsim.military.combat` reads
    ``nation.doctrine_signal`` and applies doctrine-level overrides *on top
    of* the WarAI's tactical recommendation (e.g. ``TOTAL_WAR`` forces all
    divisions to attack regardless of the WarAI output).

The per-fifth pipeline entry point :func:`worldsim.military.command.issue_doctrine`
combines this module with the fleet FSM in :mod:`worldsim.military.command`.
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Sequence, TYPE_CHECKING

from ..ai import NelderMeadPolicy

if TYPE_CHECKING:
    from ..nations.nation import Nation


class DoctrineSignal(str, Enum):
    """Strategic stance emitted by :class:`DoctrineAI` each fifth.

    Add new values here + one row in :data:`DOCTRINE_FSM_BIAS` +
    one entry in :data:`_DIVISION_DOCTRINE_OVERRIDE`.
    """
    OFFENSIVE         = "offensive"
    DEFENSIVE         = "defensive"
    ECONOMIC          = "economic"
    STRATEGIC_RESERVE = "strategic_reserve"
    TOTAL_WAR         = "total_war"



# Ordered list — append-only so existing saved models stay valid.
DOCTRINE_LIST: List[str] = [d.value for d in DoctrineSignal]

_DOCTRINE_INDEX: Dict[str, int] = {d: i for i, d in enumerate(DOCTRINE_LIST)}


def doctrine_one_hot(signal: str) -> List[float]:
    """Return a 5-dim one-hot vector for ``signal`` (index into DOCTRINE_LIST)."""
    vec = [0.0] * len(DOCTRINE_LIST)
    idx = _DOCTRINE_INDEX.get(signal, _DOCTRINE_INDEX[DoctrineSignal.DEFENSIVE.value])
    vec[idx] = 1.0
    return vec


# Division order overrides.  None → WarAI decides (possibly with bias).
_DIVISION_DOCTRINE_OVERRIDE: Dict[str, Optional[str]] = {
    DoctrineSignal.TOTAL_WAR.value:         "attack",
    DoctrineSignal.STRATEGIC_RESERVE.value: "reserve",
    DoctrineSignal.OFFENSIVE.value:         None,   # WarAI decides (attack-biased)
    DoctrineSignal.DEFENSIVE.value:         None,   # WarAI decides (defend-biased)
    DoctrineSignal.ECONOMIC.value:          None,   # WarAI decides unconstrained
}


# ===========================================================================
# DoctrineAI — runs AFTER civilian and diplomacy AIs
# ===========================================================================

class DoctrineAI(NelderMeadPolicy):
    """Strategic AI that selects a :class:`DoctrineSignal` each fifth.

    Runs at the *end* of ``process_turn``, after the civilian and diplomacy
    AIs have already executed, so it can observe their outputs.

    Inputs (10 features)
    --------------------
    0  economy (log-scaled) / 10
    1  military / 100
    2  stability / 100
    3  at_war_count / 5
    4  ally_count / 5
    5  fleet_count / 10
    6  total_fleet_firepower / 1 000
    7  last_civilian_action / 17   ← civ AI output from this very fifth
    8  total_border_pressure / 10  ← shaped by diplomacy AI
    9  enemy_fleet_threat / 1 000  ← known hostile fleet strength

    Outputs (5 scores → argmax → DoctrineSignal)
    --------------------------------------------
    One score per entry in :data:`DOCTRINE_LIST`.
    """

    _N_INPUTS:  int = 10
    _N_OUTPUTS: int = len(DOCTRINE_LIST)   # 5

    def __init__(
        self,
        *,
        table_path:    str | Path | None     = None,
        epsilon:       float                  = 0.15,
        gamma:         float                  = 0.90,
        hidden_layers: Sequence[int]          = (12, 8, 6, 4),
    ) -> None:
        super().__init__(
            self._N_INPUTS,
            self._N_OUTPUTS,
            table_path     = table_path,
            epsilon        = epsilon,
            gamma          = gamma,
            hidden_layers  = hidden_layers,
        )
        # Doctrine changes strategically (not every fifth), so the expensive
        # Nelder-Mead scale optimization runs far less often than the default.
        self._optimise_every = 64

    def choose_doctrine(self, state: List[float]) -> str:
        """Return a :class:`DoctrineSignal` value string for ``state``."""
        idx = self.choose_action(state)
        idx = max(0, min(idx, len(DOCTRINE_LIST) - 1))
        return DOCTRINE_LIST[idx]

    @staticmethod
    def build_state(
        nation:  "Nation",
        nations: Dict[int, "Nation"],
    ) -> List[float]:
        """Construct the 10-feature input vector for ``nation``."""
        fleet_fp      = sum(f.total_firepower for f in nation.fleets)
        enemy_threat  = sum(
            f.total_firepower
            for nid in nation.at_war
            if nid in nations
            for f in nations[nid].fleets
        )
        border_pressure = sum(nation.border_pressure.values())

        return [
            nation.economy / 10.0,
            min(nation.military / 100.0, 10.0),
            nation.stability / 100.0,
            min(len(nation.at_war), 5) / 5.0,
            min(len(nation.alliances), 5) / 5.0,
            min(len(nation.fleets), 10) / 10.0,
            min(fleet_fp / 1000.0, 5.0),
            getattr(nation, "last_civilian_action", 0) / 17.0,
            min(border_pressure, 10.0) / 10.0,
            min(enemy_threat / 1000.0, 5.0),
        ]
