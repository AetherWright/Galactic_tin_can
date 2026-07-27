"""Ground divisions: templates, recruitment, decisions, movement and training.

Division decisions
-------------------
Every division picks one action from :data:`DIVISION_ACTIONS` once per fifth
per enemy it faces (see :func:`decide_division_actions`, called from
``worldsim.military.combat``). When PyTorch is available this is one
*batched* forward pass through the nation's
:class:`~worldsim.ai.rnn.divisions.DivisionLoRABank` — every division the
nation owns is scored in a single call, then trained in a single batched
Double-DQN step (:func:`train_division_batch`) instead of one
perceptron-update per division. Without torch, each division falls back to
its own tiny legacy posture/movement perceptron pair (:meth:`Division.decide_posture`
/ :meth:`Division.decide_movement`), producing the same reduced
attack/defend/hold_reserve (+ deploy/recall) subset of actions this system
always supported — no regression when torch is unavailable.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

from ..ai import SimplePerceptron
from ..ai.native import make_perceptron
from ..planets import PLANETS
from .logistics import _supply_throughput

if TYPE_CHECKING:
    from ..nations.nation import Nation
    from ..ai.rnn.divisions import DivisionLoRABank


# ---------------------------------------------------------------------------
# Doctrine → maximum manpower fraction of total population
# ---------------------------------------------------------------------------

_DOCTRINE_MANPOWER: Dict[str, float] = {
    "total_war":         0.05,   # extreme wartime mobilization
    "offensive":         0.008,
    "defensive":         0.007,
    "strategic_reserve": 0.015,  # peacetime standing force
    "economic":          0.002,
}
_DEFAULT_MANPOWER: float = 0.08


ORDER_TO_LABEL = {
    "attack": [1.0, 0.0, 0.0],
    "defend": [0.0, 1.0, 0.0],
    "reserve": [0.0, 0.0, 1.0],
}

# ---------------------------------------------------------------------------
# Division action space
# ---------------------------------------------------------------------------
# Append-only, like FLEET_STATE_LIST/DOCTRINE_LIST — existing saved LoRA
# banks stay valid as long as new actions are only ever added at the end.
DIVISION_ACTIONS: List[str] = [
    "attack",           # engage enemy forces on their planet
    "defend",           # hold ground defensively while engaged
    "hold_reserve",     # stand down; excluded from combat
    "fortify",          # entrench — slow passive equipment/experience gain
    "garrison_colony",  # hold an off-homeworld colony; excluded from combat,
                         # small stability bonus for the colony it's holding
    "raid",             # aggressive high-risk strike — power bonus
    "besiege",          # patient siege warfare — smaller power bonus
    "deploy_offworld",  # seek transport (or spaceport) off the homeworld
    "recall_home",      # return to the homeworld
]
DIVISION_N_ACTIONS: int = len(DIVISION_ACTIONS)
_ACTION_INDEX: Dict[str, int] = {a: i for i, a in enumerate(DIVISION_ACTIONS)}

# Orders that pull a division out of ground-combat participation entirely
# (see worldsim.military.combat._supporting).
NONCOMBAT_ORDERS = frozenset({"hold_reserve", "garrison_colony"})

# Passive division.power multiplier by chosen order — raid/besiege trade
# risk/patience for extra offensive power; fortify rewards holding ground.
_ORDER_POWER_MULT: Dict[str, float] = {
    "raid": 1.3,
    "besiege": 1.15,
    "fortify": 1.2,
}

# Nation-level doctrine signal order, mirrors worldsim.military.doctrine's
# DoctrineSignal values — duplicated (not imported) to avoid a circular
# import, same convention as worldsim.ai.rnn.roles' _DOCTRINE_LIST.
_DOCTRINE_SIGNAL_LIST = ["offensive", "defensive", "economic", "strategic_reserve", "total_war"]

# Feature width for the batched division network — see build_division_state.
DIVISION_N_INPUTS: int = 19


@dataclass(slots=True)
class DivisionTemplate:
    """Template for creating specialised divisions."""

    name: str
    base_size: int
    equipment_mod: float = 1.0
    experience_mod: float = 1.0


@dataclass(slots=True)
class Division:
    soldiers: int
    x: int
    y: int
    planet: str
    experience: float = 1.0
    equipment: float = 1.0
    template: str = "Infantry"
    doctrine: str = "Balanced"
    # Legacy per-division posture/movement perceptrons — only ever used by
    # decide_posture()/decide_movement() (the no-torch fallback path in
    # decide_division_actions). Lazily constructed on first actual use
    # rather than eagerly here: when a torch division bank is available
    # (the common case) these are never touched at all, so eagerly building
    # one native (Rust/C++) perceptron per division per posture/movement
    # head — via an FFI call, for every division ever created, including
    # under concurrent civilian-AI threads building several divisions at
    # once — is both wasted work and (empirically, via a native heap
    # corruption crash during a long concurrent multi-nation run) a real
    # stability risk this sidesteps entirely by simply not happening.
    _controller: Optional[SimplePerceptron] = field(default=None, repr=False)
    _movement_controller: Optional[SimplePerceptron] = field(default=None, repr=False)
    posture: str = "reserve"
    order: str = "reserve"
    in_transit: bool = False
    # Transient — the (state, action) this division's order was chosen from
    # this fifth, kept only long enough for train_division_batch to consume.
    _last_state: Optional[List[float]] = None
    _last_action: Optional[int] = None

    @property
    def controller(self) -> SimplePerceptron:
        """Posture perceptron — constructed on first use (see the field
        comment above)."""
        if self._controller is None:
            self._controller = make_perceptron(3, n_outputs=3)
        return self._controller

    @property
    def movement_controller(self) -> SimplePerceptron:
        """Movement perceptron — constructed on first use (see the field
        comment above)."""
        if self._movement_controller is None:
            self._movement_controller = make_perceptron(6, n_outputs=3)
        return self._movement_controller

    def decide_posture(self, context=None) -> str:
        """Legacy per-division fallback (no torch): attack/defend/reserve."""
        if context is None:
            context = [self.soldiers / 1000.0, self.experience, self.equipment]
        scores = self.controller.predict_prob(context)
        idx = scores.index(max(scores)) if isinstance(scores, list) else 0
        self.posture = ["attack", "defend", "reserve"][idx]
        return self.posture

    def decide_movement(self, context: List[float]) -> str:
        """Legacy per-division fallback (no torch). Returns 'stay'/'deploy'/'recall'."""
        scores = self.movement_controller.predict_prob(context)
        idx = scores.index(max(scores)) if isinstance(scores, list) else 0
        return ["stay", "deploy", "recall"][idx]

    @property
    def power(self) -> float:
        mult = _ORDER_POWER_MULT.get(self.order, 1.0)
        return self.soldiers * self.experience * self.equipment * mult


# ---------------------------------------------------------------------------
# Batched decision-making (torch) + legacy per-division fallback
# ---------------------------------------------------------------------------

def build_division_state(
    div: Division,
    nation: "Nation",
    enemy: "Nation",
    nations: Dict[int, "Nation"],
) -> List[float]:
    """Return the 19-feature vector fed to the batched division network.

    Combines the division's own tactical situation (soldiers/experience/
    equipment/position, mirroring what ``move_division`` always used) with
    nation-level colonial-reach signals (star count, fleet count, whether
    this division is currently off the homeworld or embarked) so the network
    can learn to trade tactical strength for interstellar force projection.
    """
    dist_enemy = min(
        ((div.x - c.x) ** 2 + (div.y - c.y) ** 2) ** 0.5
        for c in enemy.cities
    ) if enemy.cities else 999.0
    dist_spaceport = min(
        ((div.x - s.x) ** 2 + (div.y - s.y) ** 2) ** 0.5
        for s in nation.spaceports
    ) if nation.spaceports else 999.0
    on_enemy_planet = 1.0 if div.planet == enemy.planet else 0.0
    on_home_planet = 1.0 if div.planet == nation.planet else 0.0
    supply = _supply_throughput(nation, PLANETS.get(div.planet))
    pop_ratio = nation.population / max(1, enemy.population)
    embarked = 1.0 if any(div in f.divisions for f in nation.fleets) else 0.0
    off_homeworld = 1.0 - on_home_planet
    at_colony = 1.0 if (
        off_homeworld and any(c.planet == div.planet for c in nation.cities)
    ) else 0.0
    doctrine_signal = getattr(nation, "doctrine_signal", "defensive")
    doctrine_idx = (
        _DOCTRINE_SIGNAL_LIST.index(doctrine_signal)
        if doctrine_signal in _DOCTRINE_SIGNAL_LIST else 1
    ) / 4.0
    prior_attack = 1.0 if div.order == "attack" else 0.0

    return [
        div.soldiers / 1000.0,
        div.experience,
        div.equipment,
        min(2.0, dist_enemy / 1000.0),
        min(2.0, dist_spaceport / 1000.0),
        supply,
        on_enemy_planet,
        on_home_planet,
        min(2.0, pop_ratio),
        nation.military / 100.0,
        nation.stability / 100.0,
        enemy.military / 100.0,
        nation.star_count / 10.0,
        len(nation.fleets) / 10.0,
        embarked,
        off_homeworld,
        at_colony,
        doctrine_idx,
        prior_attack,
    ]


def decide_division_actions(
    nation: "Nation",
    enemy: "Nation",
    nations: Dict[int, "Nation"],
) -> None:
    """Choose (and record) one order per division nation owns, vs. ``enemy``.

    Torch path: one batched forward call through ``nation._division_bank``
    scores every division at once. Legacy fallback: each division's own tiny
    posture/movement perceptrons decide, restricted to the original
    attack/defend/hold_reserve (+ deploy/recall) subset — unchanged behaviour
    when torch is unavailable.
    """
    if not nation.divisions:
        return

    bank = getattr(nation, "_division_bank", None)
    if bank is not None:
        states = [build_division_state(div, nation, enemy, nations) for div in nation.divisions]
        actions = bank.choose_actions(states)
        for div, state, action in zip(nation.divisions, states, actions):
            action = max(0, min(action, DIVISION_N_ACTIONS - 1))
            div.order = DIVISION_ACTIONS[action]
            div._last_state = state
            div._last_action = action
        return

    # Legacy fallback — no torch: two tiny per-division perceptrons decide
    # posture (attack/defend/reserve) and movement (stay/deploy/recall).
    for div in nation.divisions:
        posture = div.decide_posture()
        div.order = "hold_reserve" if posture == "reserve" else posture

        dist_enemy = min(
            ((div.x - c.x) ** 2 + (div.y - c.y) ** 2) ** 0.5
            for c in enemy.cities
        ) if enemy.cities else 999.0
        dist_spaceport = min(
            ((div.x - s.x) ** 2 + (div.y - s.y) ** 2) ** 0.5
            for s in nation.spaceports
        ) if nation.spaceports else 999.0
        on_enemy_planet = 1.0 if div.planet == enemy.planet else 0.0
        supply = _supply_throughput(nation, PLANETS.get(div.planet))
        pop_ratio = nation.population / max(1, enemy.population)
        context = [
            dist_enemy / 1000.0,
            dist_spaceport / 1000.0,
            supply,
            on_enemy_planet,
            float(div.order == "attack"),
            min(2.0, pop_ratio),
        ]
        movement = div.decide_movement(context)
        if movement == "deploy":
            div.order = "deploy_offworld"
        elif movement == "recall":
            div.order = "recall_home"


def _try_embark_transport(div: Division, nation: "Nation") -> bool:
    """Load ``div`` onto the first idle friendly fleet at its location with
    spare capacity. Returns ``True`` on success (the division is now aboard
    a fleet and no longer in ``nation.divisions``)."""
    for fleet in nation.fleets:
        if fleet.mission != "idle" or fleet.location != div.planet:
            continue
        used = sum(d.soldiers for d in fleet.divisions)
        if used + div.soldiers <= fleet.troop_capacity:
            fleet.divisions.append(div)
            return True
    return False


def move_division(
    div: Division,
    nation: "Nation",
    enemy: "Nation",
    nations: Dict[int, "Nation"],
) -> None:
    """Execute the movement implied by ``div.order`` (already decided by
    :func:`decide_division_actions` this fifth)."""

    if div.order == "deploy_offworld" and div.planet == nation.planet:
        if div in nation.divisions and _try_embark_transport(div, nation):
            nation.divisions.remove(div)
            return

        # No transport available — fall back to the direct spaceport route.
        friendly_ports = [
            s for nid in nation.alliances
            if (ally := nations.get(nid))
            for s in ally.spaceports
            if s.planet == enemy.planet
        ]
        friendly_ports += [s for s in nation.spaceports if s.planet == enemy.planet]

        if friendly_ports:
            target = random.choice(friendly_ports)
            div.planet = enemy.planet
            div.x = target.x
            div.y = target.y
        elif enemy.spaceports:
            # Deep strike — riskier, costs experience
            target = random.choice(enemy.spaceports)
            div.planet = enemy.planet
            div.x = target.x
            div.y = target.y
            div.experience = max(0.1, div.experience * 0.85)
            div.equipment = max(0.1, div.equipment * 0.80)

    elif div.order == "recall_home" and div.planet != nation.planet:
        home_ports = [s for s in nation.spaceports if s.planet == nation.planet]
        if home_ports:
            target = random.choice(home_ports)
            div.planet = nation.planet
            div.x = target.x
            div.y = target.y


def apply_passive_order_effects(nation: "Nation") -> None:
    """Apply small, capped per-fifth effects for orders that aren't purely
    combat/movement: fortify slowly builds equipment/experience; garrisoning
    an off-homeworld colony nudges that nation's stability up slightly,
    rewarding (and mechanically reinforcing) colonial force projection."""
    garrisoned = 0
    for div in nation.divisions:
        if div.order == "fortify":
            div.equipment = min(2.0, div.equipment + 0.01)
            div.experience = min(3.0, div.experience + 0.005)
        elif div.order == "garrison_colony" and div.planet != nation.planet:
            garrisoned += 1
    if garrisoned:
        nation.stability = min(100.0, nation.stability + min(0.5, garrisoned * 0.02))


# ---------------------------------------------------------------------------
# Colonial-capability / interstellar-force-projection reward
# ---------------------------------------------------------------------------
# Weights for (star_count, off-homeworld soldier fraction, garrisoned-colony
# count) deltas — additive on top of the plain win/loss combat outcome, so
# divisions are rewarded not just for winning battles but for actions that
# extend the nation's interstellar reach.
_PROJECTION_WEIGHTS: Tuple[float, float, float] = (5.0, 3.0, 1.0)


def division_projection_snapshot(nation: "Nation") -> Tuple[float, float, float]:
    """Return ``(star_count, off_homeworld_soldier_fraction, garrisoned_colonies)``
    for ``nation`` — before/after pair consumed by :func:`compute_division_reward`."""
    total_soldiers = sum(d.soldiers for d in nation.divisions)
    off_world = sum(d.soldiers for d in nation.divisions if d.planet != nation.planet)
    off_world_frac = (off_world / total_soldiers) if total_soldiers else 0.0
    garrisoned = sum(1 for d in nation.divisions if d.order == "garrison_colony")
    return (float(nation.star_count), off_world_frac, float(garrisoned))


def compute_division_reward(
    success: float,
    before: Tuple[float, float, float],
    after: Tuple[float, float, float],
) -> float:
    """Combine the plain combat outcome (``success``, as today) with a
    force-projection bonus from the delta between two
    :func:`division_projection_snapshot` calls."""
    diff = [a - b for a, b in zip(after, before)]
    bonus = sum(w * d for w, d in zip(_PROJECTION_WEIGHTS, diff))
    return success + bonus


def train_division_batch(nation: "Nation", divisions: List[Division], reward: float) -> None:
    """Batched training step for every division in ``divisions`` that has a
    recorded (state, action) from this fifth's :func:`decide_division_actions`.

    One call trains every division's transition in a single backward pass —
    the same ``reward`` (combat outcome + force-projection bonus, see
    :func:`compute_division_reward`) is applied to every division that took
    part, mirroring how :func:`reward_divisions` always applied one
    ``success`` value to the whole engaged group.
    """
    bank = getattr(nation, "_division_bank", None)
    if bank is None:
        return
    transitions = [
        (div._last_state, div._last_action, reward, div._last_state)
        for div in divisions
        if div._last_state is not None and div._last_action is not None
    ]
    bank.train_batch(transitions)


def reward_divisions(divisions: List[Division], success: float) -> None:
    """Legacy per-division training fallback (no torch).

    ``success`` should be positive for victory and negative for defeat.
    Successful divisions are trained toward their assigned order while
    unsuccessful ones are nudged toward a neutral reserve stance.
    """

    for div in divisions:
        context = [div.soldiers / 1000.0, div.experience, div.equipment]
        if success > 0:
            label = ORDER_TO_LABEL.get(div.order, ORDER_TO_LABEL["reserve"])
        else:
            label = ORDER_TO_LABEL["reserve"]
        div.controller.train([context], [label])


# ---------------------------------------------------------------------------
# City population helpers
# ---------------------------------------------------------------------------

def _deduct_city_population(nation: "Nation", amount: int) -> None:
    """Remove *amount* people from the nation's cities, spread proportionally.

    This makes recruitment and combat casualties persistent across turns
    because city populations are the source of ``nation.population``.
    """
    if amount <= 0 or not nation.cities:
        return
    total = sum(c.population for c in nation.cities)
    if total <= 0:
        return
    remaining = amount
    for city in nation.cities:
        if remaining <= 0:
            break
        share = int(amount * city.population / max(1, total))
        share = min(share, city.population, remaining)
        city.population = max(0, city.population - share)
        remaining -= share
    # Any rounding remainder comes from the largest city
    if remaining > 0:
        largest = max(nation.cities, key=lambda c: c.population)
        largest.population = max(0, largest.population - remaining)


def _apply_city_casualties(nation: "Nation", soldier_deaths: int) -> None:
    """Translate battlefield deaths into persistent city population loss."""
    _deduct_city_population(nation, soldier_deaths)


def build_division(nation: "Nation", template_name: Optional[str] = None) -> None:
    """Create a new division for ``nation`` using ``template_name``.

    Recruits are drawn from the city populations so the cost persists across
    turns.  ``nation.population`` is also decremented within the current turn
    so downstream budget checks in the same fifth stay consistent.
    """

    if not nation.division_templates:
        return
    if template_name is None:
        template_name = next(iter(nation.division_templates))
    tmpl = nation.division_templates.get(template_name)
    if not tmpl:
        return
    size = tmpl.base_size
    if nation.population < size:
        return
    # Deduct from city populations (persistent) and the within-turn total
    _deduct_city_population(nation, size)
    nation.population = max(0, nation.population - size)
    anchor = max(nation.cities, key=lambda c: c.population) if nation.cities else None
    x      = anchor.x      if anchor else 0
    y      = anchor.y      if anchor else 0
    planet = anchor.planet if anchor else nation.planet
    nation.divisions.append(
        Division(
            size,
            x,
            y,
            planet,
            tmpl.experience_mod,
            tmpl.equipment_mod,
            template_name,
            nation.doctrine,
        )
    )

def logistic_value(population: float, k: float, manpower_frac: float) -> float:
    """Return the logistic curve value — soldiers supportable at this population."""
    # S-curve from 0 to k*manpower_frac
    midpoint = k * 0.1  # inflection point at 10% of carrying capacity
    steepness = 0.00001  # tune this
    return (k * manpower_frac) / (1 + math.exp(-steepness * (population - midpoint)))
