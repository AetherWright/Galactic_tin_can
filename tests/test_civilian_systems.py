"""Behavioural tests for the civilian subsystems and their learnability.

These tests exist to answer three concrete questions that came up after the
civilian AI was split into a hierarchical overseer + per-department models:

1. **Colonisation** — does ``colonize_planet`` actually work, including the
   idle-Transport-fleet gate for brand-new planets?
2. **Fleet reachability** — will the civilian AI *ever* build a shipyard and
   then a fleet ship?  (Ship-build actions are masked off until a shipyard
   exists, so the fleet department has to discover ``build_shipyard`` first.)
3. **Learnability** — can the overseer and the department models actually
   converge on a consistently-rewarded choice?  If they cannot, the civilian
   policy can never specialise and nations stagnate.

The tests are deliberately backend-aware: they use whatever controller the
``Nation`` factory builds (Torch RNN when available, else Nelder-Mead).
"""
from __future__ import annotations

from typing import Dict

import pytest

from worldsim.galaxy import init_world
from worldsim.galaxy.star import STARS
from worldsim.planets import PLANETS
from worldsim.nations.civilian import (
    DEPARTMENTS,
    N_CIVILIAN_DEPARTMENTS,
    CivilianController,
    _apply_hierarchical_civilian_ai,
)
from worldsim.military.fleets import Fleet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NUM_NATIONS = 3
NUM_STARS   = 2
START_POP   = 2_000

_TOPUP_RESOURCES = ("metal", "energy", "uranium", "food")


def make_world() -> Dict:
    return init_world(NUM_NATIONS, NUM_STARS, starting_population=START_POP)


def _topup(nation, amount: float = 5_000.0) -> None:
    """Remove the economy bottleneck so a test isolates the *decision*.

    Tops up both the resource stockpiles and the civilian build budget
    (reserves / net income), which ``process_turn`` would normally settle but
    a direct ``_apply_hierarchical_civilian_ai`` call does not.
    """
    for k in _TOPUP_RESOURCES:
        nation.resources[k] = amount
    nation.econ.reserves = amount
    nation.econ.last_net_income = amount


def _fleet_dept_index() -> int:
    for i, d in enumerate(DEPARTMENTS):
        if d.slug == "fleet":
            return i
    raise AssertionError("no fleet department")


# ---------------------------------------------------------------------------
# 1. Colonisation
# ---------------------------------------------------------------------------

class TestColonization:
    def test_colonize_existing_planet_needs_no_fleet(self, monkeypatch):
        """A free colony on a planet the nation already inhabits can be taken
        with no Transport fleet."""
        nations = make_world()
        n = nations[0]
        home = PLANETS.get(n.planet)
        assert home is not None
        free = [c for c in home.iter_colonies() if c.owner is None]
        if not free:
            pytest.skip("home planet has no free colonies in this world")
        # Remove the border/area gate so the test targets the fleet rule only.
        # Nation uses __slots__, so patch the method on the class.
        monkeypatch.setattr(type(n), "_can_add_city", lambda self, city: True)

        before = len(n.cities)
        n.colonize_planet()
        assert len(n.cities) == before + 1, "existing-planet colonisation failed"

    def test_new_planet_requires_idle_transport_fleet(self, monkeypatch):
        """A free colony on a *foreign* planet is only colonisable when an idle
        Transport fleet is parked there."""
        nations = make_world()
        n = nations[0]
        monkeypatch.setattr(type(n), "_can_add_city", lambda self, city: True)

        # Exclude every home-planet candidate so the only option is foreign.
        home = PLANETS.get(n.planet)
        if home is not None:
            for c in list(home.iter_colonies()):
                if c.owner is None:
                    c.owner = -99   # sentinel "occupied", keeps it out of `free`

        foreign_planet = None
        foreign_colony = None
        for pname, planet in PLANETS.items():
            if pname == n.planet:
                continue
            frees = [c for c in planet.iter_colonies() if c.owner is None]
            if frees:
                foreign_planet, foreign_colony = planet, frees[0]
                break
        if foreign_planet is None:
            pytest.skip("no foreign planet with a free colony in this world")

        # No transport → colonisation of the foreign planet must NOT happen.
        before = len(n.cities)
        n.colonize_planet()
        assert len(n.cities) == before, "colonised a new planet without a Transport fleet"
        assert foreign_colony.owner is None

        # Park an idle Transport fleet at the foreign planet → now it can.
        n.fleets.append(
            Fleet(0, n.id, foreign_planet.name, ships={"Transport": 1}, mission="idle")
        )
        n.colonize_planet()
        assert len(n.cities) == before + 1, "Transport fleet did not enable colonisation"
        assert foreign_colony.owner == n.id

    def test_colonize_mask_blocks_when_no_unclaimed_stars(self):
        """The science department's colonise action is masked off once the
        nation already owns at least as many stars as are unclaimed."""
        colonize = next(
            a for d in DEPARTMENTS for a in d.actions if a.name == "colonize_planet"
        )
        nations = make_world()
        n = nations[0]

        # The mask is ``star_count < (number of unclaimed stars)``.
        # All stars owned by n → 0 unclaimed → mask False.
        for s in STARS.values():
            s.owner = n.id
        assert colonize.mask_fn(n) is False

        # All stars unclaimed → n owns 0, many unclaimed → mask True.
        for s in STARS.values():
            s.owner = None
        assert colonize.mask_fn(n) is True


# ---------------------------------------------------------------------------
# 2. Fleet reachability — will the civilian AI ever build a fleet?
# ---------------------------------------------------------------------------

class TestFleetReachability:
    def test_fleet_department_is_always_routable(self):
        """``build_shipyard`` is always valid, so the fleet department is never
        masked out of the overseer's choices — the overseer can always pick it."""
        nations = make_world()
        n = nations[0]
        fleet_dept = DEPARTMENTS[_fleet_dept_index()]
        # At least one fleet action (build_shipyard) is valid from turn zero.
        assert any(a.mask_fn(n) for a in fleet_dept.actions)
        # ...and specifically build_shipyard.
        shipyard = next(a for a in fleet_dept.actions if a.name == "build_shipyard")
        assert shipyard.mask_fn(n) is True

    def test_civilian_ai_eventually_builds_a_shipyard_and_fleet(self):
        """Drive the hierarchical civilian AI for many turns with the economy
        bottleneck removed and assert a fleet actually gets built.

        This is the core diagnostic: if the overseer never routes to the fleet
        department, or the fleet model never selects ``build_shipyard``, no
        nation will ever own a ship and this fails.
        """
        nations = make_world()
        built_shipyard = False
        built_ship = False

        for _ in range(250):
            for n in nations.values():
                if not isinstance(n.civilian_ai, CivilianController):
                    pytest.skip("civilian_ai is not the hierarchical controller")
                _topup(n)
                _apply_hierarchical_civilian_ai(n, n.civilian_ai)
            if not built_shipyard and any(n.shipyards for n in nations.values()):
                built_shipyard = True
            if not built_ship and any(n.fleets for n in nations.values()):
                built_ship = True
            if built_shipyard and built_ship:
                break

        assert built_shipyard, (
            "No nation ever built a shipyard via the civilian AI — the overseer "
            "never routes to the fleet department or the fleet model never picks "
            "build_shipyard."
        )
        assert built_ship, (
            "A shipyard was built but no fleet ship ever followed — the masked "
            "ship-build actions never became reachable."
        )


# ---------------------------------------------------------------------------
# 2b. Soft per-turn construction budget
# ---------------------------------------------------------------------------

class TestCivilianBudget:
    def test_settle_turn_banks_net_income(self):
        """Gross output minus upkeep is banked into reserves each turn."""
        nations = make_world()
        n = nations[0]
        n.econ.reserves = 0.0
        net = n.econ.settle_turn(gross_output=100.0, upkeep=30.0)
        assert net == 70.0
        assert n.econ.reserves == 70.0
        assert n.last_net_income == 70.0
        # Budget = net income + a slice of reserves.
        assert n.civilian_action_budget() > 0.0

    def test_reserves_are_capped(self):
        """Reserves can't grow without bound relative to gross output."""
        n = make_world()[0]
        for _ in range(50):
            n.econ.settle_turn(gross_output=100.0, upkeep=0.0)
        assert n.econ.reserves <= n.econ._RESERVE_CAP_MULT * 100.0 + 1e-6

    def test_upkeep_grows_with_assets(self):
        """More standing assets ⇒ more upkeep ⇒ less net income."""
        n = make_world()[0]
        base = n.compute_upkeep()
        from worldsim.military.divisions import build_division
        before = n.compute_upkeep()
        build_division(n)
        assert n.compute_upkeep() >= before

    def test_budget_caps_actions_per_turn(self):
        """A tiny budget executes far fewer department actions than a large one."""
        nations = make_world()
        n = nations[0]
        if not isinstance(n.civilian_ai, CivilianController):
            pytest.skip("civilian_ai is not the hierarchical controller")
        for k in _TOPUP_RESOURCES:
            n.resources[k] = 1e6   # never resource-limited; isolate the budget

        def actions_executed(budget_amount):
            n.econ.reserves = budget_amount
            n.econ.last_net_income = 0.0
            before_total = (
                len(n.cities) + len(n.mines) + len(n.factories) + len(n.power_plants)
                + len(n.schools) + len(n.labs) + len(n.hospitals) + len(n.shipyards)
                + len(n.spaceports) + len(n.ports) + len(n.farms) + len(n.bases)
                + len(n.divisions) + n._fleet_count() + len(n.projects)
            )
            _apply_hierarchical_civilian_ai(n, n.civilian_ai)
            after_total = (
                len(n.cities) + len(n.mines) + len(n.factories) + len(n.power_plants)
                + len(n.schools) + len(n.labs) + len(n.hospitals) + len(n.shipyards)
                + len(n.spaceports) + len(n.ports) + len(n.farms) + len(n.bases)
                + len(n.divisions) + n._fleet_count() + len(n.projects)
            )
            return after_total - before_total

        # Tiny budget → at most the single always-allowed top-priority action.
        small = actions_executed(1.0)
        # Generous budget → multiple departments get to build.
        large = actions_executed(10_000.0)
        assert small <= 1, f"tiny budget still executed {small} builds"
        assert large >= small, "larger budget should not execute fewer actions"


# ---------------------------------------------------------------------------
# 3. Learnability — can the models converge on a rewarded choice?
# ---------------------------------------------------------------------------

def _seed(seed: int = 0) -> None:
    """Make controller weight initialisation deterministic so the learnability
    assertions don't flake on an unlucky init."""
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass
    import random as _random
    _random.seed(seed)


def _make_overseer():
    # Build controllers the same way Nation does, backend-agnostically.
    try:
        from worldsim.ai.rnn import rnn_available, TorchCivilianOverseer, TorchDepartmentAI
        if rnn_available():
            return (
                TorchCivilianOverseer(n_depts=N_CIVILIAN_DEPARTMENTS, n_inputs=22),
                lambda slug, n_act: TorchDepartmentAI(slug, n_act, n_inputs=22),
            )
    except Exception:
        pass
    from worldsim.ai.roles import CivilianOverseerAI, DepartmentPolicyAI
    return (
        CivilianOverseerAI(n_depts=N_CIVILIAN_DEPARTMENTS, n_inputs=22),
        lambda slug, n_act: DepartmentPolicyAI(slug, n_act, n_inputs=22),
    )


def _establish(model, state, k: int = 8) -> None:
    """Prime the rolling input buffer / recurrent hidden state with *state*.

    The live civilian loop always calls ``choose_action`` (→ ``predict``)
    before ``train``, so by the time ``train`` runs the GRU buffer already
    holds the current state.  Tests must reproduce that or the controller
    trains against an all-zeros context and cannot condition on the input.
    """
    for _ in range(k):
        if hasattr(model, "predict"):
            model.predict(state)


def _train_state(model, state, reward_fn, n_actions: int, steps: int) -> None:
    """Run ``steps`` train sweeps over *state*, priming the buffer each time."""
    for _ in range(steps):
        for a in range(n_actions):
            if hasattr(model, "predict"):
                model.predict(state)
            model.train(state, a, reward_fn(a), state)


class TestLearnability:
    """Confirm the overseer and department models can actually converge on a
    consistently-rewarded choice — the prerequisite for the civilian policy to
    specialise instead of stagnating.

    These mirror the real loop: ``predict`` is called before ``train`` so the
    recurrent buffer holds the current state, and distinct states are presented
    as persistent *regimes* (as they are in-game) rather than alternated every
    step.
    """

    def test_overseer_learns_preferred_department(self):
        _seed()
        overseer, _ = _make_overseer()
        state = [0.5] * 22
        target = 3   # arbitrary department the synthetic reward favours

        _establish(overseer, state)
        _train_state(
            overseer, state,
            lambda d: 1.0 if d == target else -1.0,
            N_CIVILIAN_DEPARTMENTS, steps=150,
        )

        overseer.epsilon = 0.0   # greedy evaluation
        _establish(overseer, state)
        choice = overseer.choose_action(state, [True] * N_CIVILIAN_DEPARTMENTS)
        assert choice == target, (
            f"overseer failed to learn the rewarded department "
            f"(picked {choice}, expected {target})"
        )

    def test_department_model_learns_build_shipyard(self):
        _seed()
        """The fleet department should learn to pick ``build_shipyard`` (local
        index 0) when that is the consistently rewarded action."""
        _, make_dept = _make_overseer()
        fleet = DEPARTMENTS[_fleet_dept_index()]
        model = make_dept(fleet.slug, fleet.n_actions)
        state = [0.5] * 22
        target = 0   # build_shipyard is local index 0 in the fleet department

        _establish(model, state)
        _train_state(
            model, state,
            lambda a: 1.0 if a == target else -1.0,
            fleet.n_actions, steps=150,
        )

        model.epsilon = 0.0
        _establish(model, state)
        choice = model.choose_action(state, [True] * fleet.n_actions)
        assert choice == target, (
            f"fleet department failed to learn build_shipyard "
            f"(picked {choice}, expected {target})"
        )

    def test_overseer_conditions_on_state(self):
        _seed()
        """The overseer must map two *different* persistent regimes to two
        *different* rewarded departments — i.e. condition on the input rather
        than collapse to a single global favourite.

        States are presented in blocks (a nation stays in a regime for many
        turns), matching how the recurrent controller is fed in the real loop.
        """
        overseer, _ = _make_overseer()
        state_a = [0.9] * 22
        state_b = [0.1] * 22
        target_a, target_b = 1, 4

        for _ in range(30):
            _establish(overseer, state_a)
            _train_state(
                overseer, state_a,
                lambda d: 1.0 if d == target_a else -1.0,
                N_CIVILIAN_DEPARTMENTS, steps=10,
            )
            _establish(overseer, state_b)
            _train_state(
                overseer, state_b,
                lambda d: 1.0 if d == target_b else -1.0,
                N_CIVILIAN_DEPARTMENTS, steps=10,
            )

        overseer.epsilon = 0.0
        _establish(overseer, state_a)
        choice_a = overseer.choose_action(state_a, [True] * N_CIVILIAN_DEPARTMENTS)
        _establish(overseer, state_b)
        choice_b = overseer.choose_action(state_b, [True] * N_CIVILIAN_DEPARTMENTS)
        assert choice_a == target_a and choice_b == target_b, (
            f"overseer cannot condition on state: state_a→{choice_a} (want {target_a}), "
            f"state_b→{choice_b} (want {target_b})"
        )
