"""Civilian AI controller — standalone functions for the domestic policy loop.

All functions take a ``Nation`` instance as their first parameter, following
the same pattern used in ``war.py`` and ``diplomacy.py``.  The ``Nation``
class keeps thin wrapper methods so existing call sites are unaffected.

Action space
------------
The 22 civilian actions are organised into six *departments*.  Each
department is its own sub-model: an :class:`CivilianController` overseer
picks which department handles a given turn (a 6-way choice), and a
dedicated per-department model then makes the fine-grained choice within
that department's local action sub-space.

Each department is backed by a distinct model — for the PyTorch backend a
separate shared base model keyed ``"civilian_{slug}"`` (see
``TorchDepartmentAI``), for the Nelder-Mead backend a separate
``DepartmentPolicyAI`` — so departments specialise independently while
nations within a department still share a backbone.  The overseer and every
department model are trained on the same per-turn reward signal.

The legacy flat ``DomesticPolicyAI`` / ``TorchDomesticPolicyAI`` (selecting
from all 22 actions via their global indices) is still supported as a
fallback when a bare policy is passed in place of a ``CivilianController``.

Department layout
-----------------
    interior   (4)  — build_city, hospital, school, upgrade_assets
    industry   (5)  — mine, port, factory, power_plant, farm
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
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

from ..core import distance_sq
from ..planets import PLANETS
from ..military.divisions import build_division
from .projects import PROJECT_CATALOG, PROJECT_NAMES, NationalProject
from ..galaxy.star import STARS

if TYPE_CHECKING:
    from .nation import Nation


__all__ = [
    "DEPARTMENTS",
    "ACTION_GROUPS",
    "N_CIVILIAN_ACTIONS",
    "N_CIVILIAN_DEPARTMENTS",
    "ActionEntry",
    "CivilianDepartment",
    "CivilianController",
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
        ActionEntry(21, "build_farm",        _always_valid, lambda n: n.build_farm()),
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

# Sanity-check at import time: all global indices must be covered exactly once.
assert set(_ACTION_BY_IDX.keys()) == set(range(N_CIVILIAN_ACTIONS)), (
    f"Civilian action indices must cover 0–{N_CIVILIAN_ACTIONS - 1} exactly once "
    f"across all departments (found {sorted(_ACTION_BY_IDX.keys())})"
)


# ---------------------------------------------------------------------------
# CivilianController — hierarchical overseer + per-department sub-models
# ---------------------------------------------------------------------------

class CivilianController:
    """Hierarchical civilian AI: overseer routes to per-department sub-models.

    The overseer selects which department handles the current turn (6-way
    choice); the chosen department's model then selects a local action within
    that department's sub-space (3–5 choices).  Both models are trained on
    the same reward signal after each decision.

    All sub-models share the same 20-feature state vector as the old flat
    model.  Sub-model local indices map to global action indices via
    ``CivilianDepartment.actions[local_idx].idx``.

    Persistence
    -----------
    ``save_table`` / ``load_table`` accept a *base* path (e.g.
    ``nation_0_civilian.yml``) and derive sub-paths automatically::

        nation_0_civilian_overseer.yml
        nation_0_civilian_interior.yml
        nation_0_civilian_industry.yml
        ...
    """

    def __init__(
        self,
        overseer:        Any,
        dept_models:     Dict[str, Any],
        base_table_path: "Optional[str | Path]" = None,
    ) -> None:
        self.overseer    = overseer
        self.dept_models = dept_models           # slug → model
        self._base_path  = Path(base_table_path) if base_table_path else None
        if self._base_path:
            self.load_table(self._base_path)

    # ------------------------------------------------------------------
    # MetaGA integration
    # ------------------------------------------------------------------

    def sync_meta_ga(self, ga: object) -> None:
        """Propagate MetaGA fitness state to the overseer and every dept model."""
        for m in (self.overseer, *self.dept_models.values()):
            if hasattr(m, "sync_meta_ga"):
                m.sync_meta_ga(ga)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_table(self, path: "Optional[str | Path]" = None) -> None:
        target = Path(path) if path else self._base_path
        if target is None:
            return
        self._base_path = target
        stem, suffix = target.stem, target.suffix
        self.overseer.save_table(target.with_name(f"{stem}_overseer{suffix}"))
        for slug, model in self.dept_models.items():
            model.save_table(target.with_name(f"{stem}_{slug}{suffix}"))

    def load_table(self, path: "str | Path") -> None:
        target = Path(path)
        stem, suffix = target.stem, target.suffix
        self.overseer.load_table(target.with_name(f"{stem}_overseer{suffix}"))
        for slug, model in self.dept_models.items():
            model.load_table(target.with_name(f"{stem}_{slug}{suffix}"))

    # ------------------------------------------------------------------
    # Training helper for queued actions (global index → both models)
    # ------------------------------------------------------------------

    def train_on_global_action(
        self,
        state:      List[float],
        global_idx: int,
        reward:     float,
        new_state:  List[float],
    ) -> None:
        """Train overseer + dept model given a global action index.

        Used by ``process_action_queue`` where only the global index is known.
        """
        for dept_idx, dept in enumerate(DEPARTMENTS):
            for local_idx, action in enumerate(dept.actions):
                if action.idx == global_idx:
                    self.overseer.train(state, dept_idx, reward, new_state)
                    self.dept_models[dept.slug].train(state, local_idx, reward, new_state)
                    return


# ---------------------------------------------------------------------------
# State vector — 22 features
# ---------------------------------------------------------------------------
# Features 0–19: original economic/military/diplomatic signals.
# Feature 20: food_ratio — planet food / 20.0 (signals food pressure to the AI
#             so it can learn to prioritise build_farm when food is running low).
# Feature 21: farm count — number of owned farms.
# ---------------------------------------------------------------------------

def _civilian_state(nation: "Nation") -> List[float]:
    planet = PLANETS.get(nation.planet)
    food_ratio = min(1.0, planet.resources.get("food", 20.0) / 20.0) if planet else 1.0
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
        food_ratio,
        float(len(nation.farms)),
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

def _overseer_priority(
    ctrl: CivilianController, state: List[float], dept_valid: List[bool]
) -> List[int]:
    """Return valid department indices ordered by overseer preference.

    The overseer scores all departments for *state*; valid departments are
    returned highest-priority first.  Calling ``predict`` here also primes the
    overseer's recurrent buffer with *state* so the subsequent ``train`` calls
    operate on the correct context.  Falls back to natural order if the model
    has no ``predict`` (defensive).
    """
    valid_idx = [i for i in range(len(DEPARTMENTS)) if dept_valid[i]]
    predict = getattr(ctrl.overseer, "predict", None)
    if predict is None:
        return valid_idx
    scores = predict(state)
    valid_idx.sort(
        key=lambda i: scores[i] if i < len(scores) else float("-inf"),
        reverse=True,
    )
    return valid_idx


def _apply_hierarchical_civilian_ai(nation: "Nation", ctrl: CivilianController) -> None:
    """Priority-ordered civilian turn.

    Departments do not take turns — each turn the *overseer* ranks them by
    priority and every valid department then acts once in that order.  Because
    each build checks resources, the high-priority departments spend first and
    lower-priority ones get whatever is left (resource prioritisation rather
    than a single winner).

    Every department model **and** the overseer train each turn — the overseer
    learns a per-department value (which shapes the priority order) and no
    department is starved of training data the way single-routing did.
    """
    state0 = _civilian_state(nation)

    # A department is valid iff it has at least one currently valid action.
    dept_valid = [
        any(a.mask_fn(nation) for a in dept.actions)
        for dept in DEPARTMENTS
    ]
    if not any(dept_valid):
        return

    order = _overseer_priority(ctrl, state0, dept_valid)

    dept_rewards: Dict[int, float] = {}
    first = True
    for dept_idx in order:
        dept       = DEPARTMENTS[dept_idx]
        dept_model = ctrl.dept_models[dept.slug]
        local_valid = [a.mask_fn(nation) for a in dept.actions]
        if not any(local_valid):
            continue

        s         = _civilian_state(nation)
        local_idx = dept_model.choose_action(s, local_valid)
        global_idx = dept.actions[local_idx].idx
        if first:
            nation.last_civilian_action = global_idx
            first = False
        _execute_civilian_action(nation, global_idx)

        ns     = _civilian_state(nation)
        reward = nation.compute_reward("civilian", s, ns)
        dept_model.train(s, local_idx, reward, ns)
        dept_rewards[dept_idx] = reward

    # Train the overseer on every department it ranked this turn, using the
    # reward that department's action actually produced.  This makes the
    # overseer's per-department Q-values (and therefore the priority order)
    # track which departments pay off in the current state.
    new_state = _civilian_state(nation)
    for dept_idx, reward in dept_rewards.items():
        ctrl.overseer.train(state0, dept_idx, reward, new_state)


def _apply_civilian_ai(nation: "Nation") -> None:
    ctrl = nation.civilian_ai
    if isinstance(ctrl, CivilianController):
        _apply_hierarchical_civilian_ai(nation, ctrl)
        return

    # Legacy flat model.
    state      = _civilian_state(nation)
    valid_mask = _valid_action_mask(nation)
    idx        = ctrl.choose_action(state, valid_mask)
    nation.last_civilian_action = idx
    _execute_civilian_action(nation, idx)
    new_state  = _civilian_state(nation)
    reward     = nation.compute_reward("civilian", state, new_state)
    ctrl.train(state, idx, reward, new_state)


def process_action_queue(nation: "Nation", limit: int = 2) -> None:
    # Re-check the queue each iteration rather than pre-computing the count:
    # _execute_civilian_action may itself enqueue further actions, so a stale
    # length could skip items or pop from an exhausted queue.  We process at
    # most ``limit`` actions per call regardless of mid-loop growth.
    processed = 0
    ctrl = nation.civilian_ai
    while nation.action_queue and processed < limit:
        idx       = nation.action_queue.pop(0)
        state     = _civilian_state(nation)
        _execute_civilian_action(nation, idx)
        new_state = _civilian_state(nation)
        reward    = nation.compute_reward("civilian", state, new_state)
        if isinstance(ctrl, CivilianController):
            ctrl.train_on_global_action(state, idx, reward, new_state)
        else:
            ctrl.train(state, idx, reward, new_state)
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
        if isinstance(nation.civilian_ai, CivilianController):
            _apply_hierarchical_civilian_ai(nation, nation.civilian_ai)
        else:
            state      = _civilian_state(nation)
            valid_mask = _valid_action_mask(nation)
            idx        = nation.civilian_ai.choose_action(state, valid_mask)
            _execute_civilian_action(nation, idx)
            new_state  = _civilian_state(nation)
            reward     = nation.compute_reward("civilian", state, new_state)
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
