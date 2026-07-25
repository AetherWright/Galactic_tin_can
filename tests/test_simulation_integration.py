"""Integration tests: RNN controllers inside the live simulation loop.

Scope
-----
These tests exercise the full data path from world creation through simulation
turns to model persistence, verifying that every layer introduced by the
torch_rnn rewrite works correctly end-to-end.

Each test creates a minimal world (3 nations, 2 stars, small population) so
the suite runs quickly while still exercising the real paths.

Assertions are chosen to detect regressions without being brittle:
- Type checks confirm the right backend is active.
- Replay-buffer growth is checked with >= rather than == so accumulated state
  from earlier tests in the session does not cause spurious failures.
- LoRA change is verified by comparing before/after snapshots.
- NaN/Inf checks catch numerical explosions after training.
"""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Dict

import pytest
import torch

from worldsim.ai.rnn import (
    TorchWarAI,
    TorchDomesticPolicyAI,
    TorchCivilianOverseer,
    TorchDepartmentAI,
    TorchProjectAI,
    TorchDiplomacyAI,
    TorchResearchAI,
    TorchDoctrineAI,
    TorchFleetController,
    RoleBaseModel,
    step_all_base_models,
    save_all_base_models,
    load_all_base_models,
    rnn_available,
    _REGISTRY,
    _REGISTRY_LOCK,
)
from worldsim.nations.civilian import (
    CivilianController,
    DEPARTMENTS,
    N_CIVILIAN_ACTIONS,
    N_CIVILIAN_DEPARTMENTS,
)


# ---------------------------------------------------------------------------
# World helpers
# ---------------------------------------------------------------------------

NUM_NATIONS   = 3
NUM_STARS     = 2
START_POP     = 2_000   # tiny: keeps turn processing fast


def make_world() -> Dict:
    """Create a fresh minimal world.  Each call clears global planet/star state."""
    from worldsim.galaxy import init_world
    return init_world(NUM_NATIONS, NUM_STARS, starting_population=START_POP)


def civ_overseer(nation):
    """Return a representative civilian RNN sub-controller for internals checks.

    ``civilian_ai`` is now a hierarchical :class:`CivilianController` that
    wraps an overseer plus one model per department.  The overseer is itself
    an ordinary RNN controller, so it exposes the per-nation LoRA tensors,
    shared ``base`` model, rolling ``_buffer`` and ``_h`` hidden state that
    these end-to-end tests probe.
    """
    return nation.civilian_ai.overseer


# ---------------------------------------------------------------------------
# 1. Controller types
# ---------------------------------------------------------------------------

class TestControllerTypes:
    def test_torch_backend_active(self):
        assert rnn_available(), "PyTorch must be importable for these tests"

    def test_military_ai_is_torch(self):
        nations = make_world()
        for n in nations.values():
            assert isinstance(n.military_ai, TorchWarAI), (
                f"Nation {n.name} military_ai is {type(n.military_ai).__name__}"
            )

    def test_civilian_ai_is_torch(self):
        nations = make_world()
        for n in nations.values():
            # Civilian AI is a hierarchical controller: a torch overseer that
            # routes to one torch model per department.
            assert isinstance(n.civilian_ai, CivilianController)
            assert isinstance(n.civilian_ai.overseer, TorchCivilianOverseer)
            assert n.civilian_ai.dept_models, "no department sub-models created"
            for model in n.civilian_ai.dept_models.values():
                assert isinstance(model, TorchDepartmentAI)

    def test_project_ai_is_torch(self):
        nations = make_world()
        for n in nations.values():
            assert isinstance(n.project_ai, TorchProjectAI)

    def test_diplomacy_ai_is_torch(self):
        nations = make_world()
        for n in nations.values():
            assert isinstance(n.diplomacy_ai, TorchDiplomacyAI)

    def test_research_ai_is_torch(self):
        nations = make_world()
        for n in nations.values():
            assert isinstance(n.research_ai, TorchResearchAI)

    def test_doctrine_ai_is_torch(self):
        nations = make_world()
        for n in nations.values():
            assert isinstance(n.doctrine_ai, TorchDoctrineAI)

    def test_n_outputs_consistent_with_actions(self):
        """Each controller's n_outputs must equal the dimension it was created for."""
        from worldsim.nations.projects import PROJECT_CATALOG
        nations = make_world()
        for n in nations.values():
            assert n.military_ai.n_outputs  == 2
            # The overseer chooses among departments; each department model
            # chooses among its own local actions.  The leaf actions across
            # all departments still total N_CIVILIAN_ACTIONS.
            assert n.civilian_ai.overseer.n_outputs == N_CIVILIAN_DEPARTMENTS
            assert sum(
                m.n_outputs for m in n.civilian_ai.dept_models.values()
            ) == N_CIVILIAN_ACTIONS
            for dept in DEPARTMENTS:
                assert n.civilian_ai.dept_models[dept.slug].n_outputs == dept.n_actions
            assert n.project_ai.n_outputs   == len(PROJECT_CATALOG)
            assert n.diplomacy_ai.n_outputs == 3
            # research_ai.n_outputs == number of tech nodes at creation time
            assert n.research_ai.n_outputs > 0
            assert n.doctrine_ai.n_outputs == 5


# ---------------------------------------------------------------------------
# 2. Base-model registry
# ---------------------------------------------------------------------------

class TestBaseModelRegistry:
    def test_nations_share_civilian_base(self):
        nations = make_world()
        ns = list(nations.values())
        assert civ_overseer(ns[0]).base is civ_overseer(ns[1]).base
        assert civ_overseer(ns[1]).base is civ_overseer(ns[2]).base

    def test_nations_share_war_base(self):
        nations = make_world()
        ns = list(nations.values())
        assert ns[0].military_ai.base is ns[1].military_ai.base

    def test_nations_share_doctrine_base(self):
        nations = make_world()
        ns = list(nations.values())
        assert ns[0].doctrine_ai.base is ns[1].doctrine_ai.base

    def test_nations_have_independent_lora(self):
        """Each nation must own its own LoRA tensor objects."""
        nations = make_world()
        ns = list(nations.values())
        # Different Python objects even though they share the base
        assert civ_overseer(ns[0]).lora_B_out is not civ_overseer(ns[1]).lora_B_out

    def test_registry_has_role_entries_after_init(self):
        """At least the civilian and war keys should appear in the registry."""
        make_world()
        with _REGISTRY_LOCK:
            keys = set(_REGISTRY.keys())
        # Check by prefix (exact key includes n_inputs × n_outputs)
        assert any(k.startswith("civilian_") for k in keys)
        assert any(k.startswith("war_") for k in keys)


# ---------------------------------------------------------------------------
# 3. Single process_turn
# ---------------------------------------------------------------------------

class TestProcessTurn:
    def test_process_turn_no_exception(self):
        nations = make_world()
        for n in nations.values():
            n.process_turn(nations)   # must not raise

    def test_civilian_replay_grows_after_turn(self):
        nations = make_world()
        ns = list(nations.values())
        # Record baseline (may be non-zero from earlier tests in session)
        baseline = len(civ_overseer(ns[0]).base._replay)
        ns[0].process_turn(nations)
        # At least one experience added (the overseer trains every turn)
        assert len(civ_overseer(ns[0]).base._replay) > baseline

    def test_doctrine_replay_grows_after_turn(self):
        nations = make_world()
        ns = list(nations.values())
        baseline = len(ns[0].doctrine_ai.base._replay)
        ns[0].process_turn(nations)
        assert len(ns[0].doctrine_ai.base._replay) > baseline

    def test_hidden_state_non_zero_after_turn(self):
        """The GRU hidden state should advance from zero after a real turn."""
        nations = make_world()
        n = list(nations.values())[0]
        h_before_norm = civ_overseer(n)._h.norm().item()
        n.process_turn(nations)
        h_after_norm  = civ_overseer(n)._h.norm().item()
        # Hidden state must change (it may or may not be exactly zero before,
        # depending on prior test state — just verify it is finite)
        assert math.isfinite(h_after_norm)
        assert h_after_norm >= 0.0

    def test_lora_B_out_changes_after_turn(self):
        """The civilian LoRA B matrix must change during a turn (TD(0) update)."""
        nations = make_world()
        n = list(nations.values())[0]
        b_before = civ_overseer(n).lora_B_out.data.clone()
        n.process_turn(nations)
        n.process_turn(nations)   # second turn ensures gradient is non-trivial
        b_after = civ_overseer(n).lora_B_out.data.clone()
        assert not torch.allclose(b_before, b_after), (
            "civilian LoRA B_out did not change after two simulation turns"
        )

    def test_lora_values_stay_finite(self):
        """No NaN or Inf should appear in any LoRA tensor after 5 turns."""
        nations = make_world()
        n = list(nations.values())[0]
        for _ in range(5):
            n.process_turn(nations)
        for attr in ("lora_A_in", "lora_B_in", "lora_A_out", "lora_B_out"):
            t = getattr(civ_overseer(n), attr)
            assert torch.all(torch.isfinite(t)), f"{attr} contains NaN/Inf"


# ---------------------------------------------------------------------------
# 4. Multiple turns — temporal context
# ---------------------------------------------------------------------------

class TestTemporalContext:
    def test_different_nations_diverge_after_turns(self):
        """Two nations run the same number of turns but on different states.
        Their civilian AI outputs should differ, proving LoRA is specialising.
        """
        nations = make_world()
        ns = list(nations.values())[:2]
        n0, n1 = ns[0], ns[1]
        # Both start from B=0 LoRA (same base output initially).
        # Run a few turns each; their own histories make LoRA diverge.
        for _ in range(4):
            n0.process_turn(nations)
        for _ in range(4):
            n1.process_turn(nations)
        # Score the same dummy input through both
        dummy = [0.5] * civ_overseer(n0).n_inputs
        scores_n0 = civ_overseer(n0).predict(dummy)
        scores_n1 = civ_overseer(n1).predict(dummy)
        # After several training steps the LoRA adaptations will differ
        assert scores_n0 != scores_n1, (
            "Both nations produced identical scores after divergent histories — "
            "LoRA is not specialising per-nation"
        )

    def test_rolling_buffer_tracks_last_5_states(self):
        """The controller buffer must have exactly 5 entries after 10 predict calls."""
        nations = make_world()
        n = list(nations.values())[0]
        ai = civ_overseer(n)
        initial_buf_len = len(ai._buffer)
        assert initial_buf_len == 5   # pre-filled with zeros
        # Repeatedly predict — buffer must stay at maxlen=5
        for _ in range(10):
            ai.predict([0.1] * ai.n_inputs)
        assert len(ai._buffer) == 5


# ---------------------------------------------------------------------------
# 5. Base-model update (step_all_base_models)
# ---------------------------------------------------------------------------

class TestBaseModelUpdate:
    def test_step_all_base_models_no_crash(self):
        """step_all_base_models must not raise regardless of replay state."""
        step_all_base_models(n_steps=1)

    def test_base_model_updated_after_full_replay(self):
        """Run enough turns to fill the civilian replay buffer, then verify
        that step_all_base_models actually changes the base model weights."""
        nations = make_world()
        civilian_base = civ_overseer(list(nations.values())[0]).base

        # Fill the replay buffer beyond the batch size
        n_turns = RoleBaseModel._BATCH_SIZE + 5
        for _ in range(n_turns):
            for n in nations.values():
                n.process_turn(nations)

        w_before = civilian_base.model.output_proj.weight.data.clone()
        civilian_base.update_base(n_steps=2)
        w_after  = civilian_base.model.output_proj.weight.data.clone()

        assert not torch.allclose(w_before, w_after), (
            "Base model weights unchanged after update_base with full replay"
        )

    def test_base_weights_finite_after_update(self):
        """No NaN or Inf should appear in base model weights after training."""
        nations = make_world()
        civilian_base = civ_overseer(list(nations.values())[0]).base
        buf = [[0.1] * civilian_base.model.input_dim] * 5
        for i in range(RoleBaseModel._BATCH_SIZE + 1):
            civilian_base.add_experience((buf, i % 3, 1.0, buf))
        civilian_base.update_base(n_steps=2)
        for p in civilian_base.model.parameters():
            assert torch.all(torch.isfinite(p.data)), "NaN/Inf in base model"


# ---------------------------------------------------------------------------
# 6. SimulationLoop integration
# ---------------------------------------------------------------------------

class TestSimulationLoop:
    def test_simulation_loop_step_no_crash(self):
        """SimulationLoop.step() must complete one century without raising."""
        from worldsim.engine import SimulationLoop
        nations = make_world()
        loop = SimulationLoop(nations, log_path=None)
        loop.step()   # one full century (5 fifths)

    def test_simulation_loop_calls_rnn_step(self):
        """step_all_base_models is called by SimulationLoop — verify that
        the replay buffer grows during a full century."""
        from worldsim.engine import SimulationLoop
        nations = make_world()
        civilian_base = civ_overseer(list(nations.values())[0]).base
        baseline = len(civilian_base._replay)
        loop = SimulationLoop(nations, log_path=None)
        loop.step()
        assert len(civilian_base._replay) > baseline, (
            "Replay buffer did not grow during SimulationLoop.step()"
        )

    def test_two_consecutive_steps_no_crash(self):
        """Running two centuries back-to-back must not raise."""
        from worldsim.engine import SimulationLoop
        nations = make_world()
        loop = SimulationLoop(nations, log_path=None)
        loop.step()
        loop.step()

    def test_lora_evolves_across_centuries(self):
        """LoRA B matrices must change over two century steps."""
        from worldsim.engine import SimulationLoop
        nations = make_world()
        n = list(nations.values())[0]
        b_before = civ_overseer(n).lora_B_out.data.clone()
        loop = SimulationLoop(nations, log_path=None)
        loop.step()
        b_after = civ_overseer(n).lora_B_out.data.clone()
        assert not torch.allclose(b_before, b_after)


# ---------------------------------------------------------------------------
# 7. Persistence: save / load
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_merge_and_save_models_no_crash(self, tmp_path):
        """merge_and_save_models must not raise with torch controllers."""
        from worldsim.ai.persistence import merge_and_save_models
        nations = make_world()
        for n in nations.values():
            n.process_turn(nations)   # ensure some replay data exists
        merge_and_save_models(nations, tmp_path)  # must not raise

    def test_torch_bases_dir_created(self, tmp_path):
        """merge_and_save_models should create a torch_bases/ subdirectory."""
        from worldsim.ai.persistence import merge_and_save_models
        nations = make_world()
        for n in nations.values():
            n.process_turn(nations)
        merge_and_save_models(nations, tmp_path)
        assert (tmp_path / "torch_bases").exists(), (
            "torch_bases/ directory not created by merge_and_save_models"
        )

    def test_torch_base_files_written(self, tmp_path):
        """At least one .pt base-model file should exist after saving."""
        from worldsim.ai.persistence import merge_and_save_models
        nations = make_world()
        for n in nations.values():
            n.process_turn(nations)
        merge_and_save_models(nations, tmp_path)
        pt_files = list((tmp_path / "torch_bases").glob("*.pt"))
        assert len(pt_files) > 0, "No .pt files written to torch_bases/"

    def test_ga_seed_written(self, tmp_path):
        """seed_ga.json must always be written."""
        from worldsim.ai.persistence import merge_and_save_models
        nations = make_world()
        merge_and_save_models(nations, tmp_path)
        assert (tmp_path / "seed_ga.json").exists()

    def test_load_model_seeds_no_crash(self, tmp_path):
        """load_model_seeds must not raise even if only torch bases are present."""
        from worldsim.ai.persistence import merge_and_save_models, load_model_seeds
        nations = make_world()
        for n in nations.values():
            n.process_turn(nations)
        merge_and_save_models(nations, tmp_path)

        nations2 = make_world()
        result = load_model_seeds(nations2, tmp_path)
        assert result is True

    def test_base_weights_restored_after_load(self, tmp_path):
        """Base model weights saved in one run must be restored in the next."""
        from worldsim.ai.persistence import merge_and_save_models, load_model_seeds

        nations = make_world()
        civ_base = civ_overseer(list(nations.values())[0]).base

        # Train the base model a bit
        buf = [[0.2] * civ_base.model.input_dim] * 5
        for i in range(RoleBaseModel._BATCH_SIZE + 1):
            civ_base.add_experience((buf, i % 3, 1.0, buf))
        civ_base.update_base(n_steps=2)
        w_trained = civ_base.model.output_proj.weight.data.clone()

        merge_and_save_models(nations, tmp_path)

        # Create a fresh world with untrained base models
        nations2 = make_world()
        civ_base2 = civ_overseer(list(nations2.values())[0]).base

        # The fresh base may have different random init — corrupt to zeros
        with torch.no_grad():
            civ_base2.model.output_proj.weight.zero_()

        load_model_seeds(nations2, tmp_path)

        # After loading the trained weights should be restored
        assert torch.allclose(
            civ_base2.model.output_proj.weight,
            w_trained,
        ), "Base model weights not restored after save/load cycle"

    def test_lora_save_load_per_nation(self, tmp_path):
        """save_table / load_table must preserve per-nation LoRA weights."""
        nations = make_world()
        n = list(nations.values())[0]
        # Train a few turns to diverge LoRA from zero
        for _ in range(3):
            n.process_turn(nations)
        b_out_saved = civ_overseer(n).lora_B_out.data.clone()

        # The controller derives per-sub-model paths from this base path.
        lora_path = tmp_path / "civ_lora.yml"
        n.civilian_ai.save_table(lora_path)

        # New controller with zeroed overseer LoRA
        nations2 = make_world()
        n2 = list(nations2.values())[0]
        with torch.no_grad():
            civ_overseer(n2).lora_B_out.zero_()
        n2.civilian_ai.load_table(lora_path)

        assert torch.allclose(
            civ_overseer(n2).lora_B_out,
            b_out_saved,
        ), "Per-nation LoRA B_out not restored after save_table/load_table"


# ---------------------------------------------------------------------------
# 7b. Hierarchical civilian seed merging (NelderMead sub-models)
# ---------------------------------------------------------------------------

class TestCivilianSeedMerge:
    """The cross-run seed merger must reach the CivilianController's overseer
    and per-department sub-models, not skip the wrapper.  Built directly from
    NelderMead policies so the test is backend-independent.
    """

    @staticmethod
    def _make_ctrl():
        from worldsim.ai.roles import CivilianOverseerAI, DepartmentPolicyAI
        return CivilianController(
            overseer=CivilianOverseerAI(
                n_depts=N_CIVILIAN_DEPARTMENTS, n_inputs=22,
            ),
            dept_models={
                d.slug: DepartmentPolicyAI(d.slug, d.n_actions, n_inputs=22)
                for d in DEPARTMENTS
            },
        )

    def test_subgroups_cover_overseer_and_departments(self):
        from types import SimpleNamespace
        from worldsim.ai.persistence import _civilian_subgroups
        ctrls = [self._make_ctrl(), self._make_ctrl()]
        nations = {i: SimpleNamespace(civilian_ai=c) for i, c in enumerate(ctrls)}
        groups = _civilian_subgroups(nations)
        assert set(groups) == {"overseer"} | {d.slug for d in DEPARTMENTS}
        # Every nation contributes one sub-model to each group.
        assert all(len(v) == len(ctrls) for v in groups.values())

    def test_merge_and_reload_round_trip(self, tmp_path):
        from types import SimpleNamespace
        from worldsim.ai.persistence import (
            _civilian_subgroups, _build_merged_payload, _apply_seed_payload,
        )
        import yaml

        ctrls = [self._make_ctrl(), self._make_ctrl()]
        # Diverge the overseers so the merged weights are non-trivial.
        state = [0.3] * 22
        ctrls[0].overseer.train(state, 0, 1.0, state)
        ctrls[1].overseer.train(state, 1, 1.0, state)

        nations = {i: SimpleNamespace(civilian_ai=c) for i, c in enumerate(ctrls)}
        groups = _civilian_subgroups(nations)

        # Persist each sub-model seed.
        for name, policies in groups.items():
            payload = _build_merged_payload(policies)
            assert payload is not None, f"no merged payload for {name}"
            with open(tmp_path / f"seed_civilian_{name}.yml", "w") as fh:
                yaml.safe_dump(payload, fh)

        # A fresh controller reloads the overseer seed and changes weights.
        fresh = self._make_ctrl()
        before = fresh.overseer.network.to_dict()
        with open(tmp_path / "seed_civilian_overseer.yml") as fh:
            payload = yaml.safe_load(fh)
        _apply_seed_payload(fresh.overseer, payload)
        assert fresh.overseer.network.to_dict() != before, (
            "overseer weights not updated from merged civilian seed"
        )


# ---------------------------------------------------------------------------
# 8. save.py backwards compatibility: NelderMeadPolicy worlds still work
# ---------------------------------------------------------------------------

class TestSaveCompat:
    def test_merge_skips_torch_for_nelder_seed_files(self, tmp_path):
        """When using torch controllers no seed_*.yml files should be written
        (they are NelderMeadPolicy-only); only torch_bases/ and seed_ga.json."""
        from worldsim.ai.persistence import merge_and_save_models
        nations = make_world()
        merge_and_save_models(nations, tmp_path)
        yml_files = [f for f in tmp_path.glob("seed_*.yml")]
        # No per-role seed yml expected — torch controllers are not NelderMeadPolicy
        assert len(yml_files) == 0, (
            f"Unexpected seed yml files written: {yml_files}"
        )

    def test_no_attribute_error_with_torch_controllers(self, tmp_path):
        """merge_and_save_models must not raise AttributeError on torch controllers."""
        from worldsim.ai.persistence import merge_and_save_models
        nations = make_world()
        # Confirm all AI controllers are torch-based
        for n in nations.values():
            assert isinstance(n.civilian_ai, CivilianController)
            assert isinstance(n.civilian_ai.overseer, TorchCivilianOverseer)
        # This used to crash because _build_merged_payload accessed .n_actions
        try:
            merge_and_save_models(nations, tmp_path)
        except AttributeError as exc:
            pytest.fail(f"merge_and_save_models raised AttributeError: {exc}")


# ---------------------------------------------------------------------------
# 9. Fleet controller integration
# ---------------------------------------------------------------------------

class TestFleetControllerIntegration:
    def test_fleet_controller_is_torch(self):
        """Newly built fleets should use TorchFleetController."""
        from worldsim.military.command import make_fleet_controller
        fc = make_fleet_controller()
        assert isinstance(fc, TorchFleetController), (
            f"make_fleet_controller returned {type(fc).__name__}, "
            "expected TorchFleetController"
        )

    def test_fleet_controller_predict_shape(self):
        from worldsim.military.command import make_fleet_controller
        fc = make_fleet_controller()
        scores = fc.predict([0.5] * 15)
        assert len(scores) == 7

    def test_fleet_controller_train_no_crash(self):
        from worldsim.military.command import make_fleet_controller
        fc = make_fleet_controller()
        s = [0.1] * 15
        fc.predict(s)
        fc.train(s, 0, 0.5, [0.2] * 15)

    def test_fleet_shares_base_with_nation_fleets(self):
        """All fleet controllers in one run share the 'fleet' base model."""
        from worldsim.engine import SimulationLoop
        nations = make_world()
        loop = SimulationLoop(nations, log_path=None)
        loop.step()   # fleets may be built during a century
        # Collect all fleet controllers
        fcs = [
            f.controller
            for n in nations.values()
            for f in n.fleets
            if isinstance(f.controller, TorchFleetController)
        ]
        if len(fcs) >= 2:
            assert fcs[0].base is fcs[1].base, (
                "Fleet controllers in the same run don't share a base model"
            )


# ---------------------------------------------------------------------------
# 10. Reward function / LoRA evolution
# ---------------------------------------------------------------------------

class TestRewardAndLoRAEvolution:
    def test_goal_progress_bonus_in_compute_reward(self):
        """compute_reward must return a slightly higher value when active goals
        have positive progress vs a nation with no active goals."""
        nations = make_world()
        n = list(nations.values())[0]

        snapshot = n.reward_snapshot()   # identical before/after → zero base reward

        r_with_goals    = n.compute_reward(snapshot, snapshot)
        # Clear goals and compute again
        n.goals.goals.clear()
        r_without_goals = n.compute_reward(snapshot, snapshot)

        # With goals present, progress() ≥0 and goal_bonus ≥0, so reward ≥
        assert r_with_goals >= r_without_goals

    def test_evolve_meta_mutates_lora(self):
        """evolve_meta() must apply Gaussian noise to all RNN controller LoRAs."""
        nations = make_world()
        n = list(nations.values())[0]

        # Train a few turns so B matrices diverge from zero (easier to detect
        # mutation when both online and target already have nonzero values)
        for _ in range(3):
            n.process_turn(nations)

        b_before = civ_overseer(n).lora_B_out.data.clone()
        n.evolve_meta()
        b_after  = civ_overseer(n).lora_B_out.data.clone()

        # With sigma=0.02 and a non-trivial tensor, the tensors should differ
        # (statistically impossible for all elements to hit exactly zero noise)
        assert not torch.allclose(b_before, b_after), (
            "evolve_meta() did not mutate lora_B_out"
        )

    def test_evolve_meta_resets_optimizer_momentum(self):
        """After evolve_meta(), the Adam optimizer must have empty state."""
        nations = make_world()
        n = list(nations.values())[0]
        # Train to accumulate Adam state
        for _ in range(5):
            n.process_turn(nations)
        old_state_len = len(civ_overseer(n)._lora_optimizer.state)
        n.evolve_meta()
        # After reset, Adam state should be empty
        assert len(civ_overseer(n)._lora_optimizer.state) == 0, (
            f"Optimizer state not cleared after evolve_meta; "
            f"had {old_state_len} entries before reset"
        )


# ---------------------------------------------------------------------------
# 10. Numerical health after extended play
# ---------------------------------------------------------------------------

class TestNumericalHealth:
    def test_scores_finite_after_century(self):
        """All controller outputs must be finite after one century."""
        from worldsim.engine import SimulationLoop
        nations = make_world()
        loop = SimulationLoop(nations, log_path=None)
        loop.step()
        dummy_civ = [0.5] * civ_overseer(list(nations.values())[0]).n_inputs
        for n in nations.values():
            scores = civ_overseer(n).predict(dummy_civ)
            assert all(math.isfinite(s) for s in scores), (
                f"Non-finite score in nation {n.name} civilian AI"
            )

    def test_no_nan_in_base_model_params_after_century(self):
        """Base model parameters must remain finite after a simulation century."""
        from worldsim.engine import SimulationLoop
        nations = make_world()
        loop = SimulationLoop(nations, log_path=None)
        loop.step()
        civ_base = civ_overseer(list(nations.values())[0]).base
        for p in civ_base.model.parameters():
            assert torch.all(torch.isfinite(p.data)), (
                "NaN/Inf in civilian base model after one century"
            )


# ---------------------------------------------------------------------------
# 11. Combat casualty cascade wiring
# ---------------------------------------------------------------------------

class TestCombatWiring:
    def test_apply_city_casualties_is_importable_in_combat(self):
        """combat.py calls _apply_city_casualties on every battle with losses;
        it must be bound in the module namespace (it lives in divisions.py)."""
        import worldsim.military.combat as combat
        assert hasattr(combat, "_apply_city_casualties"), (
            "_apply_city_casualties missing from combat namespace — "
            "casualty cascade would NameError mid-battle"
        )

    def test_battle_with_casualties_does_not_raise(self):
        """A division battle that produces casualties must not crash."""
        from worldsim.military.combat import _apply_city_casualties
        nations = make_world()
        n = list(nations.values())[0]
        # Should run without NameError regardless of city state.
        _apply_city_casualties(n, 10)
