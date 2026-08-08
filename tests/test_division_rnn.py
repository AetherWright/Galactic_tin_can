"""Tests for the batched ground-division network and its integration.

Covers ``worldsim.ai.rnn.divisions`` (the shared stateless trunk + per-nation
LoRA bank) and ``worldsim.military.divisions``/``combat`` (the expanded
9-action space, the colonial/force-projection reward, and the WarAI training
wiring added alongside it).
"""
from __future__ import annotations

from typing import Dict, List

import pytest
import torch

from worldsim.ai.rnn.divisions import (
    DivisionLoRABank,
    DivisionTrunkModel,
    get_division_trunk,
    reset_division_trunk,
    save_division_trunk,
    load_division_trunk,
    step_division_trunk,
    _REPLAY,
)
from worldsim.military.divisions import (
    Division,
    DIVISION_ACTIONS,
    DIVISION_N_ACTIONS,
    DIVISION_N_INPUTS,
    NONCOMBAT_ORDERS,
    apply_passive_order_effects,
    build_division_state,
    compute_division_reward,
    decide_division_actions,
    division_projection_snapshot,
    move_division,
    reward_divisions,
    train_division_batch,
)

IN = DIVISION_N_INPUTS
OUT = DIVISION_N_ACTIONS


@pytest.fixture(autouse=True)
def _isolated_trunk():
    """Every test gets a fresh singleton trunk so LoRA-shape assumptions
    (fixed at 19x9 for this whole suite) never collide with leftovers."""
    reset_division_trunk()
    yield
    reset_division_trunk()


def fresh_bank(seed: int = 1) -> DivisionLoRABank:
    return DivisionLoRABank(IN, OUT, seed=seed)


def rand_state() -> List[float]:
    return [float(torch.randn(1).item()) for _ in range(IN)]


# ===========================================================================
# 1. DivisionTrunkModel / singleton
# ===========================================================================

class TestDivisionTrunk:
    def test_singleton_shared_across_banks(self):
        b1 = fresh_bank(1)
        b2 = fresh_bank(2)
        assert b1.trunk is b2.trunk

    def test_forward_output_shape(self):
        trunk = get_division_trunk(IN, OUT)
        x = torch.randn(4, IN)
        out = trunk(x)
        assert out.shape == (4, OUT)

    def test_reset_division_trunk_forces_rebuild(self):
        t1 = get_division_trunk(IN, OUT)
        reset_division_trunk()
        t2 = get_division_trunk(IN, OUT)
        assert t1 is not t2


# ===========================================================================
# 2. Batched inference
# ===========================================================================

class TestBatchedInference:
    def test_choose_actions_batch_size(self):
        bank = fresh_bank()
        states = [rand_state() for _ in range(10)]
        actions = bank.choose_actions(states)
        assert len(actions) == 10
        assert all(0 <= a < OUT for a in actions)

    def test_choose_actions_empty(self):
        bank = fresh_bank()
        assert bank.choose_actions([]) == []

    def test_choose_actions_respects_valid_mask(self):
        bank = fresh_bank()
        bank.epsilon = 1.0  # force exploration so the mask is exercised
        states = [rand_state()]
        mask = [False] * OUT
        mask[3] = True
        actions = bank.choose_actions(states, valid_masks=[mask])
        assert actions == [3]

    def test_one_forward_call_scores_many_divisions(self):
        """A single choose_actions call must handle a large batch without error."""
        bank = fresh_bank()
        states = [rand_state() for _ in range(500)]
        actions = bank.choose_actions(states)
        assert len(actions) == 500


# ===========================================================================
# 3. Batched training
# ===========================================================================

class TestBatchedTraining:
    def test_train_batch_changes_lora(self):
        bank = fresh_bank()
        before = bank.lora_B_in.data.clone()
        transitions = [(rand_state(), i % OUT, 1.0, rand_state()) for i in range(8)]
        bank.train_batch(transitions)
        assert not torch.allclose(before, bank.lora_B_in.data)

    def test_train_batch_empty_no_crash(self):
        bank = fresh_bank()
        bank.train_batch([])

    def test_train_batch_does_not_touch_trunk_gradients(self):
        """Mirrors the unified trunk's lock-free guarantee: a nation's batched
        training step must never populate .grad on the shared trunk."""
        bank = fresh_bank()
        for p in bank.trunk.parameters():
            p.grad = None
        transitions = [(rand_state(), 0, 1.0, rand_state())]
        bank.train_batch(transitions)
        for p in bank.trunk.parameters():
            assert p.grad is None

    def test_target_network_updates_at_frequency(self):
        bank = fresh_bank()
        bank.TARGET_UPDATE_FREQ = 3
        transitions = [(rand_state(), 0, 1.0, rand_state())]
        before = bank._target_A_in.data.clone()
        bank.train_batch(transitions)
        bank.train_batch(transitions)
        assert torch.allclose(before, bank._target_A_in.data)
        bank.train_batch(transitions)
        assert not torch.allclose(before, bank._target_A_in.data)

    def test_train_batch_feeds_shared_replay(self):
        bank = fresh_bank()
        n_before = len(_REPLAY)
        transitions = [(rand_state(), 0, 1.0, rand_state()) for _ in range(5)]
        bank.train_batch(transitions)
        assert len(_REPLAY) >= n_before + 5

    def test_step_division_trunk_no_crash_when_empty(self):
        step_division_trunk(n_steps=1)

    def test_step_division_trunk_updates_trunk_with_dense_replay(self):
        bank = fresh_bank()
        state = rand_state()
        transitions = [(state, 0, 1.0, state) for _ in range(200)]
        bank.train_batch(transitions)
        before = [p.clone() for p in bank.trunk.parameters()]
        step_division_trunk(n_steps=4)
        after = list(bank.trunk.parameters())
        assert any(not torch.allclose(b, a) for b, a in zip(before, after))


class TestConcurrentTraining:
    def test_multiple_nations_train_concurrently_no_exception(self):
        import threading

        banks = [fresh_bank(seed=i) for i in range(6)]
        errors: list[Exception] = []

        def worker(bank: DivisionLoRABank) -> None:
            try:
                for _ in range(10):
                    states = [rand_state() for _ in range(4)]
                    actions = bank.choose_actions(states)
                    transitions = [(s, a, 1.0, s) for s, a in zip(states, actions)]
                    bank.train_batch(transitions)
            except Exception as e:  # pragma: no cover
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(b,)) for b in banks]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors

    def test_same_bank_trained_from_two_threads_no_exception(self):
        """Regression test: worldsim.military.combat.wage_war(nation, ...)
        trains BOTH nation's and its enemy's division bank, and both sides'
        wage_war run concurrently on separate threads (worldsim.core.parallel
        pooled_map mode="thread") — so the *same* bank can get train_batch()
        called from two different nation threads at once. Before the
        per-bank lock this crashed with a RuntimeError from autograd's
        in-place version check (two concurrent backward+optimizer.step()
        calls racing on the same LoRA tensors)."""
        import threading

        bank = fresh_bank()
        errors: list[Exception] = []

        def worker() -> None:
            try:
                for _ in range(50):
                    states = [rand_state() for _ in range(4)]
                    actions = bank.choose_actions(states)
                    transitions = [(s, a, 1.0, s) for s, a in zip(states, actions)]
                    bank.train_batch(transitions)
            except Exception as e:  # pragma: no cover
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, f"Exceptions from concurrent same-bank training: {errors}"


# ===========================================================================
# 4. Persistence
# ===========================================================================

class TestPersistence:
    def test_save_load_lora_round_trip(self, tmp_path):
        bank = fresh_bank()
        transitions = [(rand_state(), 0, 1.0, rand_state())]
        bank.train_batch(transitions)
        path = tmp_path / "nation_1_divisions.lora.pt"
        bank.save_lora(path)

        bank2 = fresh_bank(seed=99)
        bank2.load_lora(path)
        assert torch.allclose(bank.lora_A_in.data, bank2.lora_A_in.data)
        assert torch.allclose(bank.lora_B_out.data, bank2.lora_B_out.data)

    def test_save_load_table_alias(self, tmp_path):
        bank = fresh_bank()
        bank.save_table(tmp_path / "nation_1_divisions")
        bank2 = fresh_bank(seed=2)
        bank2.load_table(tmp_path / "nation_1_divisions")
        assert torch.allclose(bank.lora_A_out.data, bank2.lora_A_out.data)

    def test_load_nonexistent_is_no_op(self, tmp_path):
        bank = fresh_bank()
        bank.load_lora(tmp_path / "missing.pt")  # should not raise

    def test_save_load_division_trunk_round_trip(self, tmp_path):
        bank = fresh_bank()
        bank.train_batch([(rand_state(), 0, 1.0, rand_state())])
        path = tmp_path / "torch_division_trunk.pt"
        save_division_trunk(path)
        before = [p.clone() for p in bank.trunk.parameters()]
        reset_division_trunk()
        load_division_trunk(path, IN, OUT)
        after = list(get_division_trunk(IN, OUT).parameters())
        assert all(torch.allclose(b, a) for b, a in zip(before, after))


# ===========================================================================
# 5. Mutation (evolve_meta support)
# ===========================================================================

class TestMutation:
    def test_mutate_lora_changes_params(self):
        bank = fresh_bank()
        before = bank.lora_A_in.data.clone()
        bank.mutate_lora(sigma=0.5)
        assert not torch.allclose(before, bank.lora_A_in.data)

    def test_mutate_lora_syncs_target(self):
        bank = fresh_bank()
        bank.mutate_lora(sigma=0.5)
        assert torch.allclose(bank.lora_A_in.data, bank._target_A_in.data)

    def test_reset_optimizer_replaces_instance(self):
        bank = fresh_bank()
        old = bank._optimizer
        bank.reset_optimizer()
        assert bank._optimizer is not old


# ===========================================================================
# 6. Division action space / state building (worldsim.military.divisions)
# ===========================================================================

def make_division(planet: str = "Earth", order: str = "attack") -> Division:
    return Division(soldiers=1000, x=0, y=0, planet=planet, order=order)


class _FakeCity:
    def __init__(self, planet: str, x: int = 0, y: int = 0):
        self.planet = planet
        self.population = 1000
        self.x = x
        self.y = y


class _FakeNation:
    """Minimal stand-in exposing only what build_division_state/mechanics touch."""

    def __init__(self, planet="Earth", divisions=None, cities=None):
        self.planet = planet
        self.divisions = divisions or []
        self.cities = cities or []
        self.spaceports = []
        self.fleets = []
        self.population = 100_000
        self.military = 50.0
        self.stability = 80.0
        self.infrastructure = 30.0
        self.resilience = 0.0
        self.tech_bonuses: Dict[str, float] = {}
        self.bases: list = []
        self.alliances: set = set()
        self.doctrine_signal = "defensive"
        self._division_bank = None

    @property
    def star_count(self) -> int:
        return 1


class TestActionSpace:
    def test_nine_actions_defined(self):
        assert DIVISION_N_ACTIONS == 9
        assert len(DIVISION_ACTIONS) == 9

    def test_noncombat_orders_subset_of_actions(self):
        assert NONCOMBAT_ORDERS <= set(DIVISION_ACTIONS)

    def test_power_multiplier_for_raid(self):
        div = make_division(order="raid")
        base = Division(soldiers=1000, x=0, y=0, planet="Earth", order="attack")
        assert div.power > base.power

    def test_power_multiplier_default_is_one(self):
        div = make_division(order="attack")
        assert div.power == pytest.approx(div.soldiers * div.experience * div.equipment)


class TestLazyLegacyControllers:
    """Division.controller/movement_controller (native Rust/C++ perceptrons,
    the no-torch fallback) must never be constructed eagerly — only when the
    legacy decide_posture/decide_movement path actually runs. Constructing
    them for every division regardless of whether torch is available was
    both wasted FFI work and, empirically, a heap-corruption crash risk
    under concurrent construction across nation threads."""

    def test_not_constructed_on_division_creation(self):
        div = make_division()
        assert div._controller is None
        assert div._movement_controller is None

    def test_not_constructed_by_batched_torch_decision(self):
        bank = fresh_bank()
        divisions = [make_division() for _ in range(4)]
        nation = _FakeNation(divisions=divisions)
        nation._division_bank = bank
        enemy = _FakeNation(planet="Mars", cities=[_FakeCity("Mars")])
        decide_division_actions(nation, enemy, {})
        for div in divisions:
            assert div._controller is None
            assert div._movement_controller is None

    def test_lazily_constructed_on_first_legacy_use(self):
        div = make_division()
        div.decide_posture()
        assert div._controller is not None
        assert div._movement_controller is None
        div.decide_movement([0.0] * 6)
        assert div._movement_controller is not None


class TestBuildDivisionState:
    def test_state_length_matches_constant(self):
        nation = _FakeNation()
        enemy = _FakeNation(planet="Mars", cities=[_FakeCity("Mars")])
        div = make_division()
        state = build_division_state(div, nation, enemy, {})
        assert len(state) == DIVISION_N_INPUTS

    def test_off_homeworld_flag(self):
        nation = _FakeNation(planet="Earth")
        enemy = _FakeNation(planet="Mars", cities=[_FakeCity("Mars")])
        home_div = make_division(planet="Earth")
        away_div = make_division(planet="Venus")
        home_state = build_division_state(home_div, nation, enemy, {})
        away_state = build_division_state(away_div, nation, enemy, {})
        # index 7 = on_home_planet, index 15 = off_homeworld
        assert home_state[7] == 1.0 and home_state[15] == 0.0
        assert away_state[7] == 0.0 and away_state[15] == 1.0


class TestDecideDivisionActions:
    def test_legacy_fallback_restricted_action_subset(self):
        """Without a torch bank, orders must stay within the original
        attack/defend/hold_reserve/deploy_offworld/recall_home subset."""
        nation = _FakeNation(divisions=[make_division() for _ in range(5)])
        enemy = _FakeNation(planet="Mars", cities=[_FakeCity("Mars")])
        decide_division_actions(nation, enemy, {})
        legacy_subset = {"attack", "defend", "hold_reserve", "deploy_offworld", "recall_home"}
        for div in nation.divisions:
            assert div.order in legacy_subset

    def test_torch_bank_records_last_state_and_action(self):
        bank = fresh_bank()
        divisions = [make_division() for _ in range(4)]
        nation = _FakeNation(divisions=divisions)
        nation._division_bank = bank
        enemy = _FakeNation(planet="Mars", cities=[_FakeCity("Mars")])
        decide_division_actions(nation, enemy, {})
        for div in divisions:
            assert div.order in DIVISION_ACTIONS
            assert div._last_state is not None
            assert div._last_action is not None

    def test_no_divisions_no_crash(self):
        nation = _FakeNation(divisions=[])
        enemy = _FakeNation(planet="Mars", cities=[_FakeCity("Mars")])
        decide_division_actions(nation, enemy, {})


class TestProjectionRewardAndPassiveEffects:
    def test_snapshot_reflects_offworld_soldiers(self):
        nation = _FakeNation(
            planet="Earth",
            divisions=[make_division(planet="Earth"), make_division(planet="Venus")],
        )
        star_count, off_frac, garrisoned = division_projection_snapshot(nation)
        assert off_frac == pytest.approx(0.5)

    def test_compute_division_reward_adds_projection_bonus(self):
        before = (1.0, 0.0, 0.0)
        after = (2.0, 0.5, 1.0)
        reward = compute_division_reward(1.0, before, after)
        # base success (1.0) + 5.0*1 + 3.0*0.5 + 1.0*1 = 1 + 5 + 1.5 + 1
        assert reward == pytest.approx(8.5)

    def test_compute_division_reward_no_change_equals_success(self):
        snap = (2.0, 0.5, 1.0)
        assert compute_division_reward(-1.0, snap, snap) == pytest.approx(-1.0)

    def test_fortify_builds_equipment_and_experience(self):
        div = make_division(order="fortify")
        div.equipment = 1.0
        div.experience = 1.0
        nation = _FakeNation(divisions=[div])
        apply_passive_order_effects(nation)
        assert div.equipment > 1.0
        assert div.experience > 1.0

    def test_garrison_colony_bumps_stability_when_offworld(self):
        div = make_division(planet="Venus", order="garrison_colony")
        nation = _FakeNation(planet="Earth", divisions=[div])
        nation.stability = 50.0
        apply_passive_order_effects(nation)
        assert nation.stability > 50.0

    def test_garrison_colony_no_effect_at_home(self):
        div = make_division(planet="Earth", order="garrison_colony")
        nation = _FakeNation(planet="Earth", divisions=[div])
        nation.stability = 50.0
        apply_passive_order_effects(nation)
        assert nation.stability == 50.0


class TestTrainDivisionBatch:
    def test_no_bank_is_no_op(self):
        nation = _FakeNation()
        div = make_division()
        div._last_state = rand_state()
        div._last_action = 0
        train_division_batch(nation, [div], 1.0)  # should not raise

    def test_trains_only_divisions_with_recorded_transitions(self):
        bank = fresh_bank()
        nation = _FakeNation()
        nation._division_bank = bank
        with_state = make_division()
        with_state._last_state = rand_state()
        with_state._last_action = 2
        without_state = make_division()
        before = bank.lora_B_in.data.clone()
        train_division_batch(nation, [with_state, without_state], 1.0)
        assert not torch.allclose(before, bank.lora_B_in.data)


class TestLegacyReserveCompat:
    def test_reward_divisions_handles_hold_reserve_label(self):
        """Legacy reward_divisions must not crash on the renamed 'hold_reserve'
        order string (previously just 'reserve')."""
        div = make_division(order="hold_reserve")
        reward_divisions([div], -1.0)  # should not raise
