"""The military arm: divisions, fleets, doctrine and combat resolution.

Submodules
----------
``divisions``  ground division templates, recruitment and movement
``logistics``  supply throughput, readiness, attrition, stacking maths
``combat``     order issuing and the per-fifth ground-war loop
``nuclear``    warhead production and strike resolution
``fleets``     space fleets, ship templates and space combat
``ftl``        FTL drive tiers and travel-time calculations
``doctrine``   DoctrineSignal vocabulary and the DoctrineAI
``command``    fleet FSM, fleet controllers and the issue_doctrine pipeline
``alliances``  military alliance blocs
"""

from .alliances import AllianceBloc
from .combat import issue_orders, wage_war
from .command import (
    DOCTRINE_FSM_BIAS,
    FLEET_STATE_LIST,
    FleetState,
    LayeredNetworkFleetController,
    build_fleet_features,
    fsm_execute_state,
    fsm_transition,
    issue_doctrine,
    make_fleet_controller,
    reward_fleet,
)
from .divisions import Division, DivisionTemplate, build_division
from .doctrine import DOCTRINE_LIST, DoctrineAI, DoctrineSignal, doctrine_one_hot
from .fleets import (
    Fleet,
    SHIP_TEMPLATES,
    SHIP_TEMPLATE_NAMES,
    ShipTemplate,
    build_fleet_ship,
    order_fleet_move,
    process_fleet_movement,
    resolve_space_combat,
)
from .ftl import (
    FTL_DRIVES,
    FTL_TIER_ORDER,
    FTLDrive,
    best_ftl_drive,
    fleet_travel_turns,
)
from .nuclear import (
    consider_first_strike,
    launch_first_strike,
    launch_nuclear_strike,
    produce_nuclear_weapons,
)

__all__ = [
    "AllianceBloc",
    "issue_orders",
    "wage_war",
    "DOCTRINE_FSM_BIAS",
    "FLEET_STATE_LIST",
    "FleetState",
    "LayeredNetworkFleetController",
    "build_fleet_features",
    "fsm_execute_state",
    "fsm_transition",
    "issue_doctrine",
    "make_fleet_controller",
    "reward_fleet",
    "Division",
    "DivisionTemplate",
    "build_division",
    "DOCTRINE_LIST",
    "DoctrineAI",
    "DoctrineSignal",
    "doctrine_one_hot",
    "Fleet",
    "SHIP_TEMPLATES",
    "SHIP_TEMPLATE_NAMES",
    "ShipTemplate",
    "build_fleet_ship",
    "order_fleet_move",
    "process_fleet_movement",
    "resolve_space_combat",
    "FTL_DRIVES",
    "FTL_TIER_ORDER",
    "FTLDrive",
    "best_ftl_drive",
    "fleet_travel_turns",
    "consider_first_strike",
    "launch_first_strike",
    "launch_nuclear_strike",
    "produce_nuclear_weapons",
]
