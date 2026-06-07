"""Civilian AI controller — standalone functions for the domestic policy loop.

All functions take a ``Nation`` instance as their first parameter, following
the same pattern used in ``war.py`` and ``diplomacy.py``.  The ``Nation``
class keeps thin wrapper methods so existing call sites are unaffected.

Action space
------------
The 21 civilian actions are organised into six *departments*.  Each
department is a natural sub-model boundary: eventually the current
``DomesticPolicyAI`` / ``TorchDomesticPolicyAI`` will act as an overseer
that picks which department handles a given turn, while dedicated per-
department models handle the fine-grained choice within that space.

For now the single existing model still selects from all 21 actions using
their flat global indices (0–20).  The department structure is metadata
only — no behaviour changes.

Department layout
-----------------
    interior   (4)  — build_city, hospital, school, upgrade_assets
    industry   (4)  — mine, port, factory, power_plant
    defense    (4)  — base, nuke_facility, orbital_defense, division
    fleet      (5)  — shipyard, frigate, transport, cruiser, battleship
    science    (3)  — spaceport, lab, colonize_planet
    executive  (1)  — start_project

Global indices are **stable** — renumbering breaks trained weights.
Local index within a department = position in ``CivilianDepartment.actions``.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, TYPE_CHECKING

from ..utils import distance_sq
from ..planets import PLANETS
from .war import build_division
from .projects import PROJECT_CATALOG, PROJECT_NAMES, NationalProject
from .star import STARS

if TYPE_CHECKING:
    from .nation import Nation


__all__ = [
    "DEPARTMENTS",
    "ACTION_GROUPS",
    "N_CIVILIAN_ACTIONS",
    "N_CIVILIAN_DEPARTMENTS",
    "ActionEntry",
    "CivilianDepartment",
    "process_action_queue",
]


# ---------------------------------------------------------------------------
# Action entry & department definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ActionEntry:
    """One leaf action in the civilian action space.

    ``idx``      — global action index (0–20); stable, referenced by the NN.
    ``name``     — human-readable identifier.
    ``mask_fn``  — (Nation) → bool; True when the action is currently valid.
    ``build_fn`` — (Nation) → None; executes the action.
    """
    idx:      int
    name:     str
    mask_fn:  Callable[["Nation"], bool]
    build_fn: Callable[["Nation"], None]


@dataclass(frozen=True)
class CivilianDepartment:
    """A logical group of related civilian actions and their future sub-model boundary.

    ``slug``    — short identifier used as a dict key (e.g. ``"interior"``).
    ``label``   — display name.
    ``actions`` — ordered tuple of :class:`ActionEntry` objects.

    The local index within a department is the position in ``actions``.
    Map to a global index with ``department.actions[local_idx].idx``.
    """
    slug:    str
    label:   str
    actions: Tuple[ActionEntry, ...]

    @property
    def global_indices(self) -> Tuple[int, ...]:
        """Global action indices belonging to this department."""
        return tuple(a.idx for a in self.actions)

    @property
    def n_actions(self) -> int:
        return len(self.actions)


def _always_valid(_: "Nation") -> bool:
    return True


# ---------------------------------------------------------------------------
# Department definitions
# Global indices MUST NOT be renumbered — existing NN weights depend on them.
# ---------------------------------------------------------------------------

DEPARTMENTS: Tuple[CivilianDepartment, ...] = (

    CivilianDepartment("interior", "Interior & Welfare", (
        ActionEntry(0,  "build_city",     _always_valid, lambda n: n.build_city()),
        ActionEntry(5,  "build_hospital", _always_valid, lambda n: n.build_hospital()),
        ActionEntry(7,  "build_school",   _always_valid, lambda n: n.build_school()),
        ActionEntry(19, "upgrade_assets", _always_valid, lambda n: n.upgrade_assets()),
    )),

    CivilianDepartment("industry", "Industry & Resources", (
        ActionEntry(2,  "build_mine",        _always_valid, lambda n: n.build_mine()),
        ActionEntry(3,  "build_port",        _always_valid, lambda n: n.build_port()),
        ActionEntry(4,  "build_factory",     _always_valid, lambda n: n.build_factory()),
        ActionEntry(8,  "build_power_plant", _always_valid, lambda n: n.build_power_plant()),
    )),

    CivilianDepartment("defense", "Defense", (
        ActionEntry(1,  "build_base",            _always_valid,
                    lambda n: n.build_base()),
        ActionEntry(11, "build_nuke_facility",
                    lambda n: "Nuclear Weapons" in n.tech_tree.unlocked,
                    lambda n: n.build_nuke_facility()),
        ActionEntry(12, "build_orbital_defense", _always_valid,
                    lambda n: n.build_orbital_defense()),
        ActionEntry(13, "build_division",        _always_valid,
                    lambda n: build_division(n)),
    )),

    CivilianDepartment("fleet", "Fleet & Naval", (
        ActionEntry(6,  "build_shipyard",             _always_valid,
                    lambda n: n.build_shipyard()),
        ActionEntry(14, "build_fleet_ship_frigate",   lambda n: bool(n.shipyards),
                    lambda n: n.build_fleet_ship("Frigate")),
        ActionEntry(15, "build_fleet_ship_transport", lambda n: bool(n.shipyards),
                    lambda n: n.build_fleet_ship("Transport")),
        ActionEntry(16, "build_fleet_ship_cruiser",   lambda n: bool(n.shipyards),
                    lambda n: n.build_fleet_ship("Cruiser")),
        ActionEntry(17, "build_fleet_ship_battleship", lambda n: bool(n.shipyards),
                    lambda n: n.build_fleet_ship("Battleship")),
    )),

    CivilianDepartment("science", "Science & Exploration", (
        ActionEntry(9,  "build_spaceport",  _always_valid,
                    lambda n: n.build_spaceport()),
        ActionEntry(10, "build_lab",        _always_valid,
                    lambda n: n.build_lab()),
        ActionEntry(18, "colonize_planet",
                    lambda n: n.star_count < len([s for s in STARS.values() if s.owner is None]),
                    lambda n: n.colonize_planet()),
    )),

    CivilianDepartment("executive", "Executive", (
        ActionEntry(20, "start_project", _always_valid, lambda n: n.start_project()),
    )),
)


# ---------------------------------------------------------------------------
# Derived constants — single source of truth for sizes and group slices
# ---------------------------------------------------------------------------

N_CIVILIAN_ACTIONS: int = sum(d.n_actions for d in DEPARTMENTS)       # 21
N_CIVILIAN_DEPARTMENTS: int = len(DEPARTMENTS)                          # 6

# Maps department slug → tuple of global indices; ready for overseer routing.
ACTION_GROUPS: Dict[str, Tuple[int, ...]] = {
    d.slug: d.global_indices for d in DEPARTMENTS
}

# Convenience reverse-lookup: global action index → ActionEntry.
_ACTION_BY_IDX: Dict[int, ActionEntry] = {
    a.idx: a for dept in DEPARTMENTS for a in dept.actions
}

# Sanity-check at import time: all 21 global indices must be covered exactly once.
assert N_CIVILIAN_ACTIONS == 21, (
    f"Expected 21 civilian actions across departments, got {N_CIVILIAN_ACTIONS}"
)
assert set(_ACTION_BY_IDX.keys()) == set(range(21)), (
    "Civilian action indices must cover 0–20 exactly once across all departments"
)


# ---------------------------------------------------------------------------
# State vector (unchanged — same 20 features as before)
# ---------------------------------------------------------------------------

def _civilian_state(nation: "Nation") -> List[float]:
    return [
        nation.economy,
        nation.technology.overall,
        nation.military,
        nation.infrastructure,
        nation.stability,
        float(len(nation.projects)),
        float(nation.star_count),
        nation._fleet_count(),
        float(len(nation.cities)),
        float(len(nation.divisions)),
        float(len(nation.mines)),
        float(len(nation.factories)),
        float(len(nation.schools)),
        float(len(nation.labs)),
        float(len(nation.hospitals)),
        nation.resources.get("metal", 0.0) / 100.0,
        nation.resources.get("uranium", 0.0) / 100.0,
        nation.resources.get("energy", 0.0) / 100.0,
        float(len(nation.at_war)),
        float(len(nation.alliances)),
    ]


# ---------------------------------------------------------------------------
# Action dispatch & validity mask — derived from DEPARTMENTS
# ---------------------------------------------------------------------------

def _execute_civilian_action(nation: "Nation", idx: int) -> None:
    action = _ACTION_BY_IDX.get(idx)
    if action is not None:
        action.build_fn(nation)


def _valid_action_mask(nation: "Nation") -> List[bool]:
    mask = [False] * N_CIVILIAN_ACTIONS
    for dept in DEPARTMENTS:
        for action in dept.actions:
            mask[action.idx] = action.mask_fn(nation)
    return mask


# ---------------------------------------------------------------------------
# Main AI loop
# ---------------------------------------------------------------------------

def _apply_civilian_ai(nation: "Nation") -> None:
    state = _civilian_state(nation)

    valid_mask = _valid_action_mask(nation)
    idx = nation.civilian_ai.choose_action(state, valid_mask)

    nation.last_civilian_action = idx
    _execute_civilian_action(nation, idx)

    new_state = _civilian_state(nation)
    reward = nation.compute_reward("civilian", state, new_state)
    nation.civilian_ai.train(state, idx, reward, new_state)


def process_action_queue(nation: "Nation", limit: int = 2) -> None:
    # Re-check the queue each iteration rather than pre-computing the count:
    # _execute_civilian_action may itself enqueue further actions, so a stale
    # length could skip items or pop from an exhausted queue.  We process at
    # most ``limit`` actions per call regardless of mid-loop growth.
    processed = 0
    while nation.action_queue and processed < limit:
        idx = nation.action_queue.pop(0)
        state = _civilian_state(nation)
        _execute_civilian_action(nation, idx)
        new_state = _civilian_state(nation)
        reward = nation.compute_reward("civilian", state, new_state)
        nation.civilian_ai.train(state, idx, reward, new_state)
        processed += 1


def _random_civilian_actions(
    nation: "Nation", nations: Dict[int, "Nation"]
) -> None:
    """Fallback actions using AI when available, otherwise random."""
    if random.random() < 0.05:
        nation.build_city()
    if nation.cities and nation.at_war:
        vulnerable = []
        for c in nation.cities:
            risk = (50 - nation.stability) / 100 if nation.stability < 50 else 0.0
            for eid in nation.at_war:
                enemy = nations.get(eid)
                if not enemy or enemy.planet != nation.planet:
                    continue
                dists_sq = [
                    distance_sq(c.coords, (d.x, d.y))
                    for d in enemy.divisions
                    if d.planet == nation.planet
                ]
                if enemy.cities:
                    dists_sq += [distance_sq(c.coords, ec.coords) for ec in enemy.cities]
                if dists_sq and min(dists_sq) < 75 * 75:
                    risk += 0.3
            if risk > 0 and random.random() < risk:
                vulnerable.append(c)
        if vulnerable:
            victim = min(vulnerable, key=lambda ct: ct.population)
            victim.owner = None
            if victim in nation.cities:
                nation.cities.remove(victim)

    if nation.civilian_ai is not None:
        state = _civilian_state(nation)
        valid_mask = _valid_action_mask(nation)
        idx = nation.civilian_ai.choose_action(state, valid_mask)
        _execute_civilian_action(nation, idx)
        new_state = _civilian_state(nation)
        reward = nation.compute_reward("civilian", state, new_state)
        nation.civilian_ai.train(state, idx, reward, new_state)
        return

    if random.random() < 0.03:
        nation.build_base()
    if random.random() < 0.02:
        nation.build_mine()
    if random.random() < 0.02:
        nation.build_port()
    if random.random() < 0.02:
        nation.build_factory()
    if random.random() < 0.01:
        nation.build_hospital()
    if random.random() < 0.01:
        nation.build_shipyard()
    if random.random() < 0.01:
        nation.build_school()
    if random.random() < 0.01:
        nation.build_power_plant()
    if random.random() < 0.01:
        nation.build_spaceport()
    if random.random() < 0.01:
        nation.build_nuke_facility()
    if random.random() < 0.01:
        nation.build_orbital_defense()
    if random.random() < 0.03:
        build_division(nation)
    if nation.spaceports and random.random() < 0.01:
        nation.colonize_planet()
    if random.random() < 0.01:
        nation.build_lab()
    if random.random() < 0.01:
        nation.start_project()
