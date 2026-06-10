"""Fleet command: the doctrine-biased FSM and fleet controllers.

FleetFSM
    Each fleet holds a lightweight neural controller
    (:class:`LayeredNetworkFleetController`, or the Torch MLP from
    :mod:`worldsim.ai.torch_fleet` when PyTorch is available).  Every fifth
    ``fsm_transition`` is called:

    1. Build a 15-feature context vector for the fleet.
    2. Ask the controller for raw per-state scores.
    3. Multiply scores by per-doctrine bias weights from
       :data:`DOCTRINE_FSM_BIAS` (easy to tune — just edit the table).
    4. ε-greedy select a :class:`FleetState`.
    5. Execute the state's action (move toward enemy, retreat, escort, etc.).
    6. Train the controller using the TD(0) target from the last reward.

Modifying the system
--------------------
* **New doctrine**  — add a value to :class:`~worldsim.military.doctrine.DoctrineSignal`,
  extend :data:`DOCTRINE_FSM_BIAS` and ``_DIVISION_DOCTRINE_OVERRIDE``, then
  update ``DOCTRINE_LIST``.
* **New fleet state** — add a value to :class:`FleetState`, handle it in
  :func:`fsm_execute_state`, add a column to :data:`DOCTRINE_FSM_BIAS`.
* **Better fleet controller** — implement ``predict`` + ``train`` on any class
  and pass it as ``Fleet.controller``; the FSM is controller-agnostic.
"""
from __future__ import annotations

import random
from collections import deque
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple, TYPE_CHECKING

from ..ai import LayeredNetwork
from ..core import distance
from .doctrine import (
    DoctrineAI,
    DoctrineSignal,
    DOCTRINE_LIST,
    _DOCTRINE_INDEX,
    doctrine_one_hot,
)

if TYPE_CHECKING:
    from ..nations.nation import Nation
    from .fleets import Fleet


class FleetState(str, Enum):
    """Operational state a fleet can occupy.

    Add new values here + handle in :func:`fsm_execute_state` +
    add a column in :data:`DOCTRINE_FSM_BIAS`.
    """
    PATROL      = "patrol"       # idle; hold position
    ASSAULT     = "assault"      # move toward nearest enemy
    DEFEND      = "defend"       # return to home planet / star
    ESCORT      = "escort"       # guard a friendly transport fleet
    RETREAT     = "retreat"      # disengage; move toward home
    REPOSITION  = "reposition"   # move to nearest unclaimed star



# Ordered list — append-only so existing saved models stay valid.
FLEET_STATE_LIST: List[str] = [s.value for s in FleetState]

_STATE_INDEX: Dict[str, int] = {s: i for i, s in enumerate(FLEET_STATE_LIST)}

# Dimension constants
_FLEET_CTRL_N_INPUTS:  int = 15
_FLEET_CTRL_N_OUTPUTS: int = len(FLEET_STATE_LIST)   # 6
_FLEET_CTRL_HIDDEN: Tuple[int, ...] = (28, 20, 14, 9)


# ===========================================================================
# Doctrine → FSM bias table
# ===========================================================================
# Each row is a doctrine; each column is a fleet state.
# The raw neural scores are multiplied by these weights before argmax.
# Higher weight → more likely under that doctrine.
#
# ADDING A NEW STATE: extend each row with one more float.
# ADDING A NEW DOCTRINE: add a row here + an entry in _DIVISION_DOCTRINE_OVERRIDE.

DOCTRINE_FSM_BIAS: Dict[str, Dict[str, float]] = {
    # fmt: off
    DoctrineSignal.OFFENSIVE.value: {
        "patrol": 0.5, "assault": 2.2, "defend": 0.4, "escort": 0.5, "retreat": 0.05, "reposition": 1.1,
    },
    DoctrineSignal.DEFENSIVE.value: {
        "patrol": 1.2, "assault": 0.3, "defend": 2.0, "escort": 1.0, "retreat": 0.7,  "reposition": 0.4,
    },
    DoctrineSignal.ECONOMIC.value: {
        "patrol": 1.5, "assault": 0.2, "defend": 0.9, "escort": 2.2, "retreat": 0.5,  "reposition": 0.5,
    },
    DoctrineSignal.STRATEGIC_RESERVE.value: {
        "patrol": 2.0, "assault": 0.1, "defend": 1.5, "escort": 0.4, "retreat": 0.3,  "reposition": 0.4,
    },
    DoctrineSignal.TOTAL_WAR.value: {
        "patrol": 0.05, "assault": 3.5, "defend": 0.2, "escort": 0.2, "retreat": 0.0, "reposition": 0.6,
    },
    # fmt: on
}


# ===========================================================================
# Fleet neural controller (pure Python, always available)
# ===========================================================================

class LayeredNetworkFleetController:
    """Fleet-level controller backed by :class:`~worldsim.ai.LayeredNetwork`.

    Feature vector (15 inputs)
    --------------------------
    0   ships_count / 20
    1   hp_ratio  (always 1.0 until per-fleet damage tracking is added)
    2   troop_load_ratio
    3   ftl_cooldown / 5
    4-8 doctrine one-hot  (5 dims)
    9   local_enemy_threat / 1 000
    10  local_ally_support / 1 000
    11  home_distance / 100
    12  nearest_enemy_distance / 100
    13  nation.stability / 100
    14  nation.military / 500

    Output (6 scores)
    -----------------
    One raw score per :class:`FleetState` entry in :data:`FLEET_STATE_LIST`.
    """

    GAMMA:      float = 0.92
    EPSILON:    float = 0.12
    MEMORY_CAP: int   = 32
    HEBBIAN_LR: float = 0.008

    def __init__(
        self,
        n_inputs:  int = _FLEET_CTRL_N_INPUTS,
        n_outputs: int = _FLEET_CTRL_N_OUTPUTS,
        *,
        seed: Optional[int] = None,
    ) -> None:
        self.n_inputs  = n_inputs
        self.n_outputs = n_outputs
        self.network   = LayeredNetwork(
            n_inputs,
            [*_FLEET_CTRL_HIDDEN, n_outputs],
            seed=seed,
        )
        self.scales: List[float] = [1.0] * sum(
            len(layer) for layer in self.network.weights
        )
        self.memory: Deque[Tuple[List[float], List[float]]] = deque(
            maxlen=self.MEMORY_CAP
        )
        self._step: int = 0

    # ------------------------------------------------------------------

    def _pad(self, inputs: List[float]) -> List[float]:
        vec = list(inputs)[: self.n_inputs]
        vec += [0.0] * max(0, self.n_inputs - len(vec))
        return vec

    def predict(self, inputs: List[float]) -> List[float]:
        """Forward pass; returns one raw score per FSM state."""
        raw = self.network.forward(self._pad(inputs), self.scales)
        if isinstance(raw, tuple):
            raw = raw[0]
        return [float(v) for v in raw]

    def train(
        self,
        state:      List[float],
        action:     int,
        reward:     float,
        next_state: List[float],
    ) -> None:
        """Online TD(0) update."""
        if not (0 <= action < self.n_outputs):
            return
        s  = self._pad(state)
        ns = self._pad(next_state)
        current = self.predict(s)
        future  = self.predict(ns)
        target  = list(current)
        target[action] = reward + self.GAMMA * max(future)
        self.memory.append((s, target))
        result = self.network.forward(s, self.scales, return_trace=True)
        if isinstance(result, tuple) and self.HEBBIAN_LR > 0:
            _, trace = result
            if trace:
                self.network.apply_hebbian(trace, self.HEBBIAN_LR)
        self._step += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type":      "layered_network_fleet_controller",
            "n_inputs":  self.n_inputs,
            "n_outputs": self.n_outputs,
            "network":   self.network.to_dict(),
            "scales":    list(self.scales),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "LayeredNetworkFleetController":
        obj = cls(
            n_inputs  = int(payload.get("n_inputs",  _FLEET_CTRL_N_INPUTS)),
            n_outputs = int(payload.get("n_outputs", _FLEET_CTRL_N_OUTPUTS)),
        )
        nd = payload.get("network")
        if isinstance(nd, dict):
            obj.network = LayeredNetwork.from_dict(nd)
        scales = payload.get("scales")
        if isinstance(scales, list):
            obj.scales = [float(v) for v in scales]
        return obj


def make_fleet_controller() -> LayeredNetworkFleetController:
    """Return the best available fleet controller.

    Prefers the Torch fleet controller from :mod:`worldsim.ai.torch_fleet` when PyTorch is
    importable; otherwise returns :class:`LayeredNetworkFleetController`.
    """
    try:
        from ..ai.torch_fleet import TorchFleetController
        if TorchFleetController is not None:
            return TorchFleetController()  # type: ignore[return-value]
    except (ImportError, AttributeError):
        pass
    return LayeredNetworkFleetController()


# ===========================================================================
# Fleet feature extraction
# ===========================================================================

def _resolve_coords(location: str) -> Tuple[float, float, float]:
    """Return 3-D coords for planet or star named ``location``, or origin."""
    try:
        from ..galaxy.star import STARS
        star = STARS.get(location)
        if star:
            return (star.x, star.y, star.z)
    except Exception:
        pass
    try:
        from ..planets import PLANETS
        planet = PLANETS.get(location)
        if planet:
            return planet.coords
    except Exception:
        pass
    return (0.0, 0.0, 0.0)


def build_fleet_features(
    fleet:           "Fleet",
    nation:          "Nation",
    nations:         Dict[int, "Nation"],
    doctrine_signal: str,
) -> List[float]:
    """Return the 15-float feature vector fed to the fleet controller.

    Index  Meaning
    -----  -------
    0      ships_count / 20
    1      hp_ratio (placeholder 1.0 until per-fleet damage is tracked)
    2      troop_load_ratio
    3      ftl_cooldown / 5
    4-8    doctrine one-hot (5 values)
    9      local enemy fleet firepower / 1 000
    10     local allied fleet firepower / 1 000
    11     distance to home planet / 100
    12     distance to nearest enemy / 100
    13     nation stability / 100
    14     nation military / 500
    """
    # --- fleet stats ---
    ships_norm   = min(fleet.total_ships / 20.0, 5.0)
    troop_ratio  = (
        sum(d.soldiers for d in fleet.divisions) / max(1, fleet.troop_capacity)
        if fleet.troop_capacity > 0 else 0.0
    )
    cooldown_norm = min(fleet.ftl_cooldown / 5.0, 2.0)

    # --- doctrine one-hot ---
    doc_vec = doctrine_one_hot(doctrine_signal)

    # --- spatial / threat ---
    my_coords   = _resolve_coords(fleet.location)
    home_coords = _resolve_coords(nation.planet)
    home_dist   = min(distance(my_coords, home_coords) / 100.0, 5.0)

    enemy_threat        = 0.0
    nearest_enemy_dist  = 5.0
    ally_support        = 0.0

    for nid, other in nations.items():
        if nid == nation.id:
            continue
        is_enemy = nid in nation.at_war
        is_ally  = nid in nation.alliances
        for of in other.fleets:
            coords = _resolve_coords(of.location)
            d = distance(my_coords, coords) / 100.0
            if is_enemy:
                enemy_threat += min(of.total_firepower / 1000.0, 2.0)
                nearest_enemy_dist = min(nearest_enemy_dist, d)
            elif is_ally:
                ally_support += min(of.total_firepower / 1000.0, 2.0)

    return [
        ships_norm,
        1.0,                        # hp_ratio placeholder
        min(troop_ratio, 1.0),
        cooldown_norm,
        *doc_vec,                   # 5 values (indices 4-8)
        min(enemy_threat, 5.0),
        min(ally_support, 5.0),
        home_dist,
        min(nearest_enemy_dist, 5.0),
        nation.stability / 100.0,
        min(nation.military / 500.0, 2.0),
    ]


# ===========================================================================
# FSM state actions
# ===========================================================================

def _nearest_enemy_target(
    fleet:   "Fleet",
    nation:  "Nation",
    nations: Dict[int, "Nation"],
) -> Optional[Tuple[str, float]]:
    """Return ``(location, distance_ly)`` of the closest enemy fleet or planet."""
    my = _resolve_coords(fleet.location)
    best: Optional[Tuple[str, float]] = None

    for nid in nation.at_war:
        other = nations.get(nid)
        if not other:
            continue
        for ef in other.fleets:
            d = distance(my, _resolve_coords(ef.location))
            if best is None or d < best[1]:
                best = (ef.location, d)
        if other.planet:
            d = distance(my, _resolve_coords(other.planet))
            if best is None or d < best[1]:
                best = (other.planet, d)
    return best


def _nearest_uncontrolled_star(
    fleet:   "Fleet",
    nations: Dict[int, "Nation"],
) -> Optional[Tuple[str, float]]:
    """Return ``(star_name, distance_ly)`` of the nearest star with no owner."""
    try:
        from ..galaxy.star import STARS
    except Exception:
        return None
    my = _resolve_coords(fleet.location)
    best: Optional[Tuple[str, float]] = None
    for star in STARS.values():
        if star.owner is not None:
            continue
        d = distance(my, (star.x, star.y, star.z))
        if best is None or d < best[1]:
            best = (star.name, d)
    return best


def _find_transport_to_escort(
    nation: "Nation",
    fleet:  "Fleet",
) -> Optional["Fleet"]:
    """Return a nearby friendly transport fleet that has troops aboard."""
    my = _resolve_coords(fleet.location)
    candidates = [
        f for f in nation.fleets
        if f.fleet_id != fleet.fleet_id
        and f.troop_capacity > 0
        and f.divisions
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda f: distance(my, _resolve_coords(f.location)),
    )


def fsm_execute_state(
    fleet:   "Fleet",
    state:   str,
    nation:  "Nation",
    nations: Dict[int, "Nation"],
) -> None:
    """Carry out the action implied by FSM ``state`` for ``fleet``.

    Only acts when the fleet is idle — never interrupts an in-progress move.
    """
    # Lazy import to break the fleet ↔ military_ai import cycle
    from .fleets import order_fleet_move

    if fleet.mission == "moving":
        return   # already underway

    if state == FleetState.PATROL.value:
        pass  # intentionally idle

    elif state == FleetState.ASSAULT.value:
        target = _nearest_enemy_target(fleet, nation, nations)
        if target:
            dest, dist = target
            order_fleet_move(nation, fleet, dest, max(dist, 0.01))

    elif state in (FleetState.DEFEND.value, FleetState.RETREAT.value):
        if fleet.location != nation.planet:
            dist = distance(
                _resolve_coords(fleet.location),
                _resolve_coords(nation.planet),
            )
            order_fleet_move(nation, fleet, nation.planet, max(dist, 0.01))

    elif state == FleetState.ESCORT.value:
        transport = _find_transport_to_escort(nation, fleet)
        if transport:
            dist = distance(
                _resolve_coords(fleet.location),
                _resolve_coords(transport.location),
            )
            if dist > 0.1:
                order_fleet_move(nation, fleet, transport.location, dist)

    elif state == FleetState.REPOSITION.value:
        target = _nearest_uncontrolled_star(fleet, nations)
        if target:
            dest, dist = target
            order_fleet_move(nation, fleet, dest, max(dist, 0.01))


# ===========================================================================
# FSM transition  (controller → doctrine bias → action)
# ===========================================================================

def fsm_transition(
    fleet:           "Fleet",
    nation:          "Nation",
    nations:         Dict[int, "Nation"],
    doctrine_signal: str,
) -> str:
    """Select and execute a new FSM state for ``fleet``.

    Steps:
    1. Build feature vector.
    2. Controller predicts raw scores.
    3. Per-doctrine bias weights applied (multiplicative).
    4. ε-greedy argmax selects new :class:`FleetState`.
    5. :func:`fsm_execute_state` carries out the action.

    Returns the chosen state string.
    """
    controller = fleet.controller
    if controller is None:
        fleet.state = FleetState.PATROL.value
        return fleet.state

    features   = build_fleet_features(fleet, nation, nations, doctrine_signal)
    raw_scores = controller.predict(features)

    # Apply per-doctrine bias
    bias_table = DOCTRINE_FSM_BIAS.get(doctrine_signal, {})
    biased = [
        raw_scores[i] * bias_table.get(FLEET_STATE_LIST[i], 1.0)
        for i in range(min(len(raw_scores), len(FLEET_STATE_LIST)))
    ]

    # ε-greedy selection
    epsilon = getattr(controller, "EPSILON", 0.12)
    if random.random() < epsilon:
        state_idx = random.randrange(len(FLEET_STATE_LIST))
    else:
        state_idx = biased.index(max(biased)) if biased else 0

    state_idx = max(0, min(state_idx, len(FLEET_STATE_LIST) - 1))
    fleet.state = FLEET_STATE_LIST[state_idx]

    fsm_execute_state(fleet, fleet.state, nation, nations)
    return fleet.state


def reward_fleet(
    fleet:         "Fleet",
    outcome:       float,
    prev_features: List[float],
) -> None:
    """Train the fleet controller based on the outcome of its last action.

    ``outcome`` > 0 indicates success (survived, arrived, won combat).
    ``outcome`` < 0 indicates failure (fleet destroyed or routed).
    """
    controller = fleet.controller
    if controller is None or not prev_features:
        return
    state_idx = _STATE_INDEX.get(fleet.state, 0)
    # For simplicity, approximate next_state ≈ current state
    if hasattr(controller, "train"):
        controller.train(prev_features, state_idx, outcome, prev_features)


# ===========================================================================
# Issue doctrine — top-level per-nation military AI entry point
# ===========================================================================

def issue_doctrine(
    nation:  "Nation",
    nations: Dict[int, "Nation"],
    *,
    alliance_row: Optional[List[int]] = None,
) -> None:
    """Run the full military AI pipeline for ``nation`` — call this AFTER civ/diplo.

    1. :class:`DoctrineAI` reads civilian action + current state → emits
       a new :class:`DoctrineSignal` stored as ``nation.doctrine_signal``.
    2. Each fleet's FSM transitions under the new doctrine.
    3. Fleet controllers are trained based on a survival-outcome proxy.

    Division orders are **not** set here — they are set by the doctrine-aware
    ``issue_orders`` call in :mod:`worldsim.military.combat` at the start of each
    fifth's combat phase, using the *stored* ``doctrine_signal``.
    """
    if not nation.doctrine_ai:
        return   # no military AI initialised; fleets remain on their current state

    # ------------------------------------------------------------------
    # 1. Generate doctrine signal
    # ------------------------------------------------------------------
    prev_signal = getattr(nation, "doctrine_signal", DoctrineSignal.DEFENSIVE.value)
    prev_state  = DoctrineAI.build_state(nation, nations)
    prev_idx    = _DOCTRINE_INDEX.get(prev_signal, _DOCTRINE_INDEX[DoctrineSignal.DEFENSIVE.value])

    new_signal  = nation.doctrine_ai.choose_doctrine(prev_state)
    nation.doctrine_signal = new_signal

    # Quick reward estimate for the doctrine AI: improvement in military + stability
    new_state   = DoctrineAI.build_state(nation, nations)
    reward      = (new_state[1] - prev_state[1]) + (new_state[2] - prev_state[2])
    nation.doctrine_ai.train(prev_state, prev_idx, reward, new_state)

    # ------------------------------------------------------------------
    # 2. Run fleet FSMs
    # ------------------------------------------------------------------
    for fleet in list(nation.fleets):
        if fleet.is_empty():
            continue
        prev_features = build_fleet_features(fleet, nation, nations, new_signal)
        fsm_transition(fleet, nation, nations, new_signal)
        # Survival proxy: fleets that still exist get a small positive reward
        outcome = 0.05 if not fleet.is_empty() else -0.5
        reward_fleet(fleet, outcome, prev_features)
