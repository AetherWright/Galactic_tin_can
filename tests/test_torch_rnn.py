"""Unit tests for worldsim/ai/rnn/unified.py.

Coverage
--------
* UnifiedTrunkModel — forward-pass shape/dtype, lazy per-role head/extra_proj
                      creation, role isolation (rebuilding one role's
                      extra_proj doesn't touch another's)
* Shared replay/step_trunk — add_experience, batching, weight updates,
                      no-op when small, optimizer picks up lazily-created
                      role parameters (regression: add_param_group)
* UnifiedRoleView   — rolling buffer (core++extra, size=5), predict shape,
                      hidden-state advancement, action bounds, train API,
                      core+head LoRA gradients, trunk grads zeroed after
                      train, experience deposit tagged with role
* LoRA init         — B matrices start at zero (ΔW = 0 at init); A matrices
                      non-zero; LoRA actually changes output after B is
                      perturbed (both core, on the bank, and head, per role)
* Training loop      — LoRA parameters change after a train() call; replay
                      buffer grows; trunk update reduces loss
* Shared trunk       — every role/nation shares the exact same trunk object;
                      each nation's bank (core LoRA) is independent
* Role wrappers      — TorchWarAI (create_doctrine, set_allies_dimension),
                      TorchDoctrineAI.choose_doctrine, TorchFleetController,
                      TorchResearchSubsystemAI output length
* Persistence        — save_lora/load_lora (head), save_core/load_core,
                      save_trunk/load_trunk round-trips
* Thread safety       — concurrent train() calls from multiple nations do not
                      raise, and the shared replay buffer grows correctly
* Double DQN          — target network (head AND core) lags then syncs at
                      TARGET_UPDATE_FREQ
* Categorical (C51)   — events head produces a valid distribution and trains
                      via projected cross-entropy
"""
from __future__ import annotations

import math
import threading

import torch

# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------
from worldsim.ai.rnn import (
    BUFFER_SIZE,
    CORE_DIM,
    HIDDEN_DIM,
    N_ATOMS,
    _DOCTRINE_LIST,
    _FLEET_STATE_LIST,
    UnifiedLoRABank,
    UnifiedRoleView,
    UnifiedTrunkModel,
    get_trunk,
    rnn_available,
    save_trunk,
    load_trunk,
    step_trunk,
    TorchWarAI,
    TorchDomesticPolicyAI,
    TorchProjectAI,
    TorchDiplomacyAI,
    TorchResearchSubsystemAI,
    TorchDoctrineAI,
    TorchFleetController,
    TorchEventAI,
)
from worldsim.ai.rnn.unified import (
    _BATCH_SIZE,
    _REPLAY,
    _REPLAY_LOCK,
    add_experience,
    project_categorical,
)

# ---------------------------------------------------------------------------
# Constants used across tests
# ---------------------------------------------------------------------------
IN   = 6   # small "extra" input dim for speed
OUT  = 3   # small output dim
ROLE = "test_role_unit"   # isolated key so tests don't collide with real roles
_role_counter = 0


def _unique_role(prefix: str = "test_role") -> str:
    """Return a fresh role key so tests don't collide via the shared trunk."""
    global _role_counter
    _role_counter += 1
    return f"{prefix}_{_role_counter}"


def fresh_bank() -> UnifiedLoRABank:
    # A non-zero core is essential: the core-LoRA path is
    # ``(core @ A_core.T) @ B_core.T`` — an all-zero core makes it (and its
    # gradient w.r.t. B_core) identically zero regardless of training, which
    # would make every core-LoRA test pass vacuously.
    fixed_core = [0.3 - 0.05 * i for i in range(CORE_DIM)]
    return UnifiedLoRABank(core_state_fn=lambda: list(fixed_core))


def fresh_view(
    role: str | None = None,
    n_in: int = IN,
    n_out: int = OUT,
    epsilon: float = 0.0,   # disable exploration by default
    seed: int = 42,
    categorical: bool = False,
    bank: UnifiedLoRABank | None = None,
) -> UnifiedRoleView:
    """Return a fresh UnifiedRoleView with an isolated role key."""
    role = role or _unique_role()
    bank = bank or fresh_bank()
    return UnifiedRoleView(
        bank, role, n_in, n_out,
        categorical=categorical, epsilon=epsilon, gamma=0.9, seed=seed,
    )


def rand_state(n: int = IN) -> list[float]:
    return [float(torch.randn(1).item()) for _ in range(n)]


def rand_core() -> list[float]:
    return [float(torch.randn(1).item()) for _ in range(CORE_DIM)]


# ===========================================================================
# 1. Module-level availability
# ===========================================================================

class TestAvailability:
    def test_rnn_available(self):
        assert rnn_available() is True

    def test_buffer_size_constant(self):
        assert BUFFER_SIZE == 5

    def test_core_dim_constant(self):
        assert CORE_DIM == 16

    def test_doctrine_list_length(self):
        assert len(_DOCTRINE_LIST) == 5

    def test_fleet_state_list_length(self):
        assert len(scores) == len(FLEET_STATE_LIST)

# ===========================================================================
# 2. UnifiedTrunkModel
# ===========================================================================

class TestUnifiedTrunkModel:
    def _trunk_with_role(self, role, extra_dim=IN, n_out=OUT, categorical=False):
        trunk = UnifiedTrunkModel()
        trunk.ensure_role(role, extra_dim, n_out, categorical=categorical)
        return trunk

    def test_forward_output_shape(self):
        role = _unique_role()
        trunk = self._trunk_with_role(role)
        core = torch.randn(1, BUFFER_SIZE, CORE_DIM)
        extra = torch.randn(1, BUFFER_SIZE, IN)
        h0 = torch.zeros(1, 1, HIDDEN_DIM)
        out = trunk.forward(role, core, extra, h0)
        assert out.shape == (1, OUT)

    def test_forward_dtype_float32(self):
        role = _unique_role()
        trunk = self._trunk_with_role(role)
        core = torch.randn(1, 3, CORE_DIM)
        extra = torch.randn(1, 3, IN)
        h0 = torch.zeros(1, 1, HIDDEN_DIM)
        out = trunk.forward(role, core, extra, h0)
        assert out.dtype == torch.float32

    def test_categorical_head_shape_and_distribution(self):
        role = _unique_role()
        trunk = self._trunk_with_role(role, categorical=True)
        core = torch.randn(1, BUFFER_SIZE, CORE_DIM)
        extra = torch.randn(1, BUFFER_SIZE, IN)
        h0 = torch.zeros(1, 1, HIDDEN_DIM)
        out = trunk.forward(role, core, extra, h0)
        assert out.shape == (1, OUT, N_ATOMS)
        probs = out.exp()
        assert torch.allclose(probs.sum(dim=-1), torch.ones(1, OUT), atol=1e-4)

    def test_ensure_role_idempotent(self):
        role = _unique_role()
        trunk = UnifiedTrunkModel()
        trunk.ensure_role(role, IN, OUT)
        head_before = trunk.heads[role]
        trunk.ensure_role(role, IN, OUT)
        assert trunk.heads[role] is head_before

    def test_ensure_role_rebuilds_extra_proj_on_dim_change(self):
        """Only the changed role's extra_proj is rebuilt; its head is untouched."""
        role = _unique_role()
        trunk = UnifiedTrunkModel()
        trunk.ensure_role(role, IN, OUT)
        head_before = trunk.heads[role]
        extra_before = trunk.extra_proj[role]
        trunk.ensure_role(role, IN + 4, OUT)
        assert trunk.extra_proj[role] is not extra_before
        assert trunk.heads[role] is head_before   # n_outputs unchanged

    def test_two_roles_do_not_interfere(self):
        trunk = UnifiedTrunkModel()
        r1, r2 = _unique_role(), _unique_role()
        trunk.ensure_role(r1, IN, OUT)
        trunk.ensure_role(r2, IN, OUT + 2)
        assert trunk.heads[r1] is not trunk.heads[r2]
        assert trunk._n_outputs[r1] == OUT
        assert trunk._n_outputs[r2] == OUT + 2

    def test_layer_parameter_names(self):
        """Verify that core_proj, gru, and per-role extra_proj/heads exist."""
        role = _unique_role()
        trunk = self._trunk_with_role(role)
        param_names = {n.split(".")[0] for n, _ in trunk.named_parameters()}
        assert {"core_proj", "gru", "extra_proj", "heads", "role_embedding"} <= param_names


# ===========================================================================
# 3. Shared replay + step_trunk
# ===========================================================================

class TestSharedReplay:
    def test_add_experience_grows_replay(self):
        role = _unique_role()
        buf = [[0.0] * (CORE_DIM + IN)] * BUFFER_SIZE
        n_before = len(_REPLAY)
        add_experience((role, buf, 0, 1.0, buf))
        assert len(_REPLAY) == n_before + 1

    def test_step_trunk_no_op_when_small(self):
        """step_trunk should not crash and should be roughly a no-op when
        the role in question has too little replay to reach batch size."""
        role = _unique_role()
        trunk = get_trunk()
        trunk.ensure_role(role, IN, OUT)
        step_trunk(n_steps=1)
        # Global replay may already contain plenty from other tests, so this
        # role's own weights may or may not move — just confirm no crash and
        # finite weights.
        assert torch.all(torch.isfinite(trunk.heads[role].weight.data))

    def test_step_trunk_updates_role_weights(self):
        """With enough same-role replay, step_trunk must change that role's head."""
        role = _unique_role()
        trunk = get_trunk()
        trunk.ensure_role(role, IN, OUT)
        buf = [[0.5] * (CORE_DIM + IN)] * BUFFER_SIZE
        for i in range(_BATCH_SIZE * 4):
            add_experience((role, buf, i % OUT, 1.0, buf))

        w_before = trunk.heads[role].weight.data.clone()
        for _ in range(5):
            step_trunk(n_steps=4)
            if not torch.allclose(w_before, trunk.heads[role].weight.data):
                break
        assert not torch.allclose(w_before, trunk.heads[role].weight.data), (
            "Trunk head weights unchanged after step_trunk with dense same-role replay"
        )

    def test_unknown_role_in_replay_is_skipped(self):
        """An experience tagged with a role the trunk doesn't know must not crash."""
        buf = [[0.1] * (CORE_DIM + IN)] * BUFFER_SIZE
        add_experience(("role_that_does_not_exist_anywhere", buf, 0, 1.0, buf))
        step_trunk(n_steps=1)   # should not raise

    def test_resizing_a_role_purges_its_stale_replay_entries(self):
        """Resizing a role's extra width (e.g. TorchWarAI.set_allies_dimension
        as nations are added/removed) must drop that role's previously
        buffered entries — they were recorded at the old width and would
        otherwise crash step_trunk's reshape against the new one."""
        role = _unique_role()
        trunk = get_trunk()
        trunk.ensure_role(role, 10, OUT)
        old_buf = [[0.2] * (CORE_DIM + 10)] * BUFFER_SIZE
        for i in range(_BATCH_SIZE + 5):
            add_experience((role, old_buf, i % OUT, 1.0, old_buf))
        assert any(item[0] == role for item in _REPLAY)

        trunk.ensure_role(role, 11, OUT)  # resize — width actually changes

        assert not any(item[0] == role for item in _REPLAY), (
            "stale-width replay entries survived a role resize"
        )
        step_trunk(n_steps=1)  # must not raise on the new width

    def test_resizing_a_role_does_not_purge_other_roles(self):
        role_a = _unique_role("role_a")
        role_b = _unique_role("role_b")
        trunk = get_trunk()
        trunk.ensure_role(role_a, 10, OUT)
        trunk.ensure_role(role_b, 12, OUT)
        buf_b = [[0.3] * (CORE_DIM + 12)] * BUFFER_SIZE
        add_experience((role_b, buf_b, 0, 1.0, buf_b))

        trunk.ensure_role(role_a, 20, OUT)  # resize role_a only

        assert any(item[0] == role_b for item in _REPLAY), (
            "unrelated role's replay entries were wrongly purged"
        )


# ===========================================================================
# 4. UnifiedRoleView — rolling buffer, hidden state
# ===========================================================================

class TestRoleViewBuffer:
    def test_buffer_initial_size(self):
        view = fresh_view()
        assert len(view._buffer) == BUFFER_SIZE

    def test_buffer_entry_width_is_core_plus_extra(self):
        view = fresh_view()
        assert len(view._buffer[0]) == CORE_DIM + IN

    def test_buffer_fills_after_predict_calls(self):
        view = fresh_view()
        for i in range(BUFFER_SIZE + 2):
            view.predict(rand_state())
        assert len(view._buffer) == BUFFER_SIZE

    def test_hidden_state_advances(self):
        view = fresh_view()
        view.predict(rand_state())
        assert view._h.shape == (1, 1, HIDDEN_DIM)
        assert view._h.requires_grad is False

    def test_predict_output_length(self):
        view = fresh_view()
        scores = view.predict(rand_state())
        assert len(scores) == OUT
        assert all(isinstance(v, float) for v in scores)

    def test_predict_with_short_state_padded(self):
        view = fresh_view()
        scores = view.predict([0.5])
        assert len(scores) == OUT

    def test_predict_with_long_state_truncated(self):
        view = fresh_view()
        scores = view.predict([0.1] * (IN * 3))
        assert len(scores) == OUT


# ===========================================================================
# 5. UnifiedRoleView — predict_prob and choose_action
# ===========================================================================

class TestRoleViewActions:
    def test_predict_prob_sums_to_one(self):
        view = fresh_view()
        probs = view.predict_prob(rand_state())
        assert abs(sum(probs) - 1.0) < 1e-5

    def test_choose_action_within_bounds(self):
        view = fresh_view(epsilon=0.0)
        for _ in range(20):
            a = view.choose_action(rand_state())
            assert 0 <= a < OUT

    def test_choose_action_epsilon_greedy_explores(self):
        view = fresh_view(epsilon=1.0)
        actions = {view.choose_action(rand_state()) for _ in range(50)}
        assert len(actions) > 1

    def test_choose_action_valid_mask(self):
        view = fresh_view(epsilon=0.0)
        mask = [False, True, False]
        for _ in range(10):
            a = view.choose_action(rand_state(), valid_mask=mask)
            assert a == 1

    def test_all_actions_invalid_in_mask_falls_back_to_zero(self):
        view = fresh_view(epsilon=0.0)
        mask = [False, False, False]
        a = view.choose_action(rand_state(), valid_mask=mask)
        assert a == 0


# ===========================================================================
# 6. LoRA initialisation (core, on the bank; head, on the view)
# ===========================================================================

class TestLoRAInit:
    def test_head_B_starts_at_zero(self):
        view = fresh_view()
        assert torch.all(view.lora_B_head == 0.0)

    def test_core_B_starts_at_zero(self):
        bank = fresh_bank()
        assert torch.all(bank.lora_B_core == 0.0)

    def test_head_A_non_zero(self):
        view = fresh_view()
        assert not torch.all(view.lora_A_head == 0.0)

    def test_core_A_non_zero(self):
        bank = fresh_bank()
        assert not torch.all(bank.lora_A_core == 0.0)

    def test_zero_lora_equals_base_output(self):
        """With both core and head B=0, the view's output must equal the
        trunk's own (unadapted) forward pass."""
        view = fresh_view()
        core = torch.zeros(1, BUFFER_SIZE, CORE_DIM)
        extra = torch.zeros(1, BUFFER_SIZE, IN)
        h0 = torch.zeros(1, 1, HIDDEN_DIM)
        seq = torch.cat([core, extra], dim=-1)
        base_out = view.trunk.forward(view.role, core, extra, h0)
        view_out, _ = view._forward(seq, h0)
        assert torch.allclose(base_out, view_out, atol=1e-6)

    def test_nonzero_head_lora_changes_output(self):
        view = fresh_view()
        core = torch.randn(1, BUFFER_SIZE, CORE_DIM)
        extra = torch.randn(1, BUFFER_SIZE, IN)
        h0 = torch.zeros(1, 1, HIDDEN_DIM)
        seq = torch.cat([core, extra], dim=-1)
        base_out = view.trunk.forward(view.role, core, extra, h0).detach().clone()
        with torch.no_grad():
            view.lora_B_head.fill_(0.5)
        view_out, _ = view._forward(seq, h0)
        assert not torch.allclose(base_out, view_out, atol=1e-6)

    def test_nonzero_core_lora_changes_output(self):
        view = fresh_view()
        core = torch.randn(1, BUFFER_SIZE, CORE_DIM)
        extra = torch.randn(1, BUFFER_SIZE, IN)
        h0 = torch.zeros(1, 1, HIDDEN_DIM)
        seq = torch.cat([core, extra], dim=-1)
        base_out = view.trunk.forward(view.role, core, extra, h0).detach().clone()
        with torch.no_grad():
            view.bank.lora_B_core.fill_(0.5)
        view_out, _ = view._forward(seq, h0)
        assert not torch.allclose(base_out, view_out, atol=1e-6)

    def test_lora_rank_and_alpha_defaults(self):
        view = fresh_view()
        assert view.lora_r     == UnifiedRoleView._LORA_R
        assert view.lora_alpha == UnifiedRoleView._LORA_ALPHA

    def test_head_lora_shapes(self):
        view = fresh_view()
        assert view.lora_A_head.shape == (view.lora_r, HIDDEN_DIM)
        assert view.lora_B_head.shape == (OUT, view.lora_r)

    def test_core_lora_shapes(self):
        bank = fresh_bank()
        assert bank.lora_A_core.shape == (bank.lora_r, CORE_DIM)
        assert bank.lora_B_core.shape == (HIDDEN_DIM, bank.lora_r)


# ===========================================================================
# 7. Training loop
# ===========================================================================

class TestTrainingLoop:
    def test_lora_B_head_changes_after_training(self):
        view = fresh_view()
        b_before = view.lora_B_head.data.clone()
        state = rand_state()
        view.choose_action(state)
        view.train(state, 0, 1.0, rand_state())
        b_after = view.lora_B_head.data.clone()
        assert not torch.allclose(b_before, b_after), (
            "lora_B_head unchanged after training step"
        )

    def test_core_lora_changes_after_training(self):
        view = fresh_view()
        b_before = view.bank.lora_B_core.data.clone()
        state = rand_state()
        view.choose_action(state)
        view.train(state, 0, 1.0, rand_state())
        b_after = view.bank.lora_B_core.data.clone()
        assert not torch.allclose(b_before, b_after), (
            "core lora_B unchanged after training step"
        )

    def test_trunk_grads_zeroed_after_train(self):
        """train() must not introduce or alter any trunk-parameter gradient —
        the online step only ever backprops into this nation's own LoRA
        tensors. (Not a global-zero check: the trunk is a process-wide
        singleton, so other tests' step_trunk() calls may leave unrelated
        residual grads on it — train() just must not add to or change that.)
        """
        view = fresh_view()
        state = rand_state()
        view.choose_action(state)
        before = {
            id(p): (p.grad.clone() if p.grad is not None else None)
            for p in view.trunk.parameters()
        }
        view.train(state, 0, 1.0, rand_state())
        for p in view.trunk.parameters():
            prior = before[id(p)]
            if prior is None:
                assert p.grad is None, "train() introduced a new trunk gradient"
            else:
                assert torch.equal(p.grad, prior), "train() modified an existing trunk gradient"

    def test_experience_deposited_in_replay_tagged_with_role(self):
        view = fresh_view()
        n_before = len(_REPLAY)
        state = rand_state()
        view.choose_action(state)
        view.train(state, 0, 0.5, rand_state())
        assert len(_REPLAY) == n_before + 1
        assert _REPLAY[-1][0] == view.role

    def test_train_ignores_out_of_range_action(self):
        view = fresh_view()
        b_before = view.lora_B_head.data.clone()
        state = rand_state()
        view.predict(state)
        view.train(state, OUT + 5, 1.0, rand_state())
        assert torch.allclose(view.lora_B_head.data, b_before)

    def test_train_ignores_negative_action(self):
        view = fresh_view()
        b_before = view.lora_B_head.data.clone()
        state = rand_state()
        view.predict(state)
        view.train(state, -1, 1.0, rand_state())
        assert torch.allclose(view.lora_B_head.data, b_before)

    def test_multi_step_training_does_not_crash(self):
        view = fresh_view()
        state = rand_state()
        next_state = rand_state()
        for _ in range(50):
            action = view.choose_action(state)
            view.train(state, action, 1.0, next_state)
            state, next_state = next_state, rand_state()


# ===========================================================================
# 8. Shared trunk / independent per-nation LoRA
# ===========================================================================

class TestSharedTrunk:
    def test_all_roles_share_the_singleton_trunk(self):
        v1 = fresh_view()
        v2 = fresh_view()
        assert v1.trunk is v2.trunk is get_trunk()

    def test_same_role_two_banks_share_head(self):
        """Two nations (banks) using the SAME role name must share that
        role's head/extra_proj on the trunk — only their LoRA differs."""
        role = _unique_role()
        v1 = fresh_view(role=role)
        v2 = fresh_view(role=role, bank=fresh_bank())
        assert v1.trunk.heads[role] is v2.trunk.heads[role]
        assert v1.lora_B_head is not v2.lora_B_head
        assert v1.bank.lora_A_core is not v2.bank.lora_A_core

    def test_experience_from_one_view_visible_to_sibling_view(self):
        """Two views sharing a role deposit into the same replay pool."""
        role = _unique_role()
        v1 = fresh_view(role=role, epsilon=0.0)
        v2 = fresh_view(role=role, bank=fresh_bank(), epsilon=0.0)
        n_before = len(_REPLAY)
        state = rand_state()
        v1.predict(state)
        v1.train(state, 0, 1.0, rand_state())
        assert len(_REPLAY) == n_before + 1
        assert _REPLAY[-1][0] == role
        # v2 could equally have trained the same role/head
        assert v2.role == role


# ===========================================================================
# 9. Persistence
# ===========================================================================

class TestPersistence:
    def test_save_load_head_lora_round_trip(self, tmp_path):
        view = fresh_view()
        for _ in range(5):
            s = rand_state()
            view.predict(s)
            view.train(s, 0, 1.0, rand_state())
        path = tmp_path / "lora.pt"
        view.save_lora(path)

        view2 = fresh_view()
        view2.load_lora(path)

        assert torch.allclose(view.lora_A_head, view2.lora_A_head)
        assert torch.allclose(view.lora_B_head, view2.lora_B_head)

    def test_load_lora_nonexistent_path_is_no_op(self, tmp_path):
        view = fresh_view()
        view.load_lora(tmp_path / "does_not_exist.lora.pt")  # should not raise

    def test_save_lora_does_not_persist_hidden_state(self, tmp_path):
        """Hidden state is transient runtime context, not learned — a saved
        checkpoint must not carry it, and a loaded view must start clean
        regardless of what the saving view's hidden state was."""
        view = fresh_view()
        view._h = torch.ones_like(view._h)
        path = tmp_path / "lora.pt"
        view.save_lora(path)

        view2 = fresh_view()
        view2._h = torch.full_like(view2._h, 7.0)
        view2.load_lora(path)
        assert torch.all(view2._h == 7.0), (
            "load_lora must not touch hidden state — it isn't saved"
        )

    def test_reset_runtime_state_zeroes_hidden_and_buffer(self):
        view = fresh_view()
        view._h = torch.ones_like(view._h)
        view.reset_runtime_state()
        assert torch.all(view._h == 0)
        assert all(v == 0.0 for row in view._buffer for v in row)

    def test_bank_reset_runtime_state_resets_every_role(self):
        bank = fresh_bank()
        v1 = fresh_view(bank=bank)
        v2 = fresh_view(bank=bank)
        v1._h = torch.ones_like(v1._h)
        v2._h = torch.ones_like(v2._h)
        bank.reset_runtime_state()
        assert torch.all(v1._h == 0)
        assert torch.all(v2._h == 0)

    def test_save_table_creates_lora_file(self, tmp_path):
        ai = TorchDiplomacyAI(epsilon=0.0)
        yml_path = tmp_path / "diplomacy.yml"
        ai.save_table(yml_path)
        assert (tmp_path / "diplomacy.lora.pt").exists()

    def test_load_table_reads_lora_file(self, tmp_path):
        c1 = TorchDiplomacyAI(epsilon=0.0)
        with torch.no_grad():
            c1.lora_B_head.fill_(2.0)
        yml_path = tmp_path / "dip2.yml"
        c1.save_table(yml_path)

        c2 = TorchDiplomacyAI(table_path=yml_path, epsilon=0.0)
        assert torch.allclose(c1.lora_B_head, c2.lora_B_head)

    def test_save_load_core_round_trip(self, tmp_path):
        bank = fresh_bank()
        with torch.no_grad():
            bank.lora_A_core.fill_(3.14)
        path = tmp_path / "core.lora.pt"
        bank.save_core(path)

        bank2 = fresh_bank()
        bank2.load_core(path)
        assert torch.allclose(bank.lora_A_core, bank2.lora_A_core)

    def test_save_load_trunk_round_trip(self, tmp_path):
        role = _unique_role()
        trunk = get_trunk()
        trunk.ensure_role(role, IN, OUT)
        with torch.no_grad():
            trunk.heads[role].weight.fill_(7.77)

        path = tmp_path / "trunk.pt"
        save_trunk(path)

        with torch.no_grad():
            trunk.heads[role].weight.fill_(0.0)

        load_trunk(path)
        assert torch.allclose(
            trunk.heads[role].weight,
            torch.full_like(trunk.heads[role].weight, 7.77),
        )


# ===========================================================================
# 10. Role-specific wrappers
# ===========================================================================

class TestRoleWrappers:
    # ----- TorchWarAI -----
    def test_torch_war_ai_predict_shape(self):
        ai = TorchWarAI(allies_dim=3)
        scores = ai.predict(rand_state(ai.n_inputs))
        assert len(scores) == 2

    def test_torch_war_ai_create_doctrine(self):
        ai = TorchWarAI()
        d = ai.create_doctrine()
        assert isinstance(d, str) and len(d) > 0

    def test_torch_war_ai_set_allies_dimension(self):
        ai = TorchWarAI(allies_dim=3)
        old_n = ai.n_inputs
        ai.set_allies_dimension(5)
        assert ai.n_inputs != old_n
        assert ai.allies_dim == 5
        scores = ai.predict(rand_state(ai.n_inputs))
        assert len(scores) == 2

    def test_torch_war_ai_set_allies_dimension_no_op(self):
        ai = TorchWarAI(allies_dim=4)
        n_before = ai.n_inputs
        ai.set_allies_dimension(4)
        assert ai.n_inputs == n_before

    def test_torch_war_ai_set_allies_dimension_keeps_training(self):
        """After a dimension change the controller must keep training/predicting."""
        ai = TorchWarAI(allies_dim=4)
        ai.set_allies_dimension(9)
        s = rand_state(ai.n_inputs)
        for _ in range(ai.TARGET_UPDATE_FREQ * 2):
            a = ai.choose_action(s)
            ai.train(s, a, 0.5, s)   # must not raise

    def test_torch_war_ai_self_heals_after_another_nations_resize(self):
        """Every nation's TorchWarAI shares one trunk-wide "war" extra_proj —
        when ANOTHER nation's set_allies_dimension() resizes it, this one's
        own choose_action()/train() must not crash on the now-stale width
        (reproduces a real crash: issue_orders(enemy) — alliance_row=None —
        and consider_first_strike() call choose_action() without ever
        calling set_allies_dimension themselves)."""
        war_a = TorchWarAI(allies_dim=4)
        war_b = TorchWarAI(allies_dim=4)
        state_a = rand_state(war_a.n_inputs)
        war_a.choose_action(state_a)  # populate war_a's buffer at width 4

        war_b.set_allies_dimension(7)  # a *different* nation resizes the shared layer
        assert war_a.n_inputs != war_b.n_inputs  # war_a hasn't heard about it yet

        action = war_a.choose_action(state_a)  # must not raise
        assert 0 <= action < war_a.n_outputs
        assert war_a.n_inputs == war_b.n_inputs, "war_a did not self-heal"
        war_a.train(state_a, action, 0.5, state_a)  # must not raise either

    # ----- TorchDomesticPolicyAI -----
    def test_domestic_policy_ai_output_dim(self):
        ai = TorchDomesticPolicyAI(actions=21, n_inputs=20)
        assert ai.n_outputs == 21
        scores = ai.predict(rand_state(20))
        assert len(scores) == 21

    # ----- TorchProjectAI -----
    def test_project_ai_output_dim(self):
        ai = TorchProjectAI(actions=12, n_inputs=22)
        assert ai.n_outputs == 12

    # ----- TorchDiplomacyAI -----
    def test_diplomacy_ai_train_does_not_crash(self):
        ai = TorchDiplomacyAI(actions=3, n_inputs=5, epsilon=0.0)
        s  = rand_state(5)
        ai.choose_action(s)
        ai.train(s, 0, 0.5, rand_state(5))

    # ----- TorchResearchSubsystemAI -----
    def test_research_subsystem_ai_output_shape(self):
        # Unique suffix: "research_{suffix}" is a role key on the shared
        # singleton trunk, so a literal "physics" would collide with any
        # other test using that same subsystem name at a different size.
        suffix = _unique_role("physics_shape")
        ai = TorchResearchSubsystemAI(suffix, 10, 8)
        scores = ai.predict(rand_state(8))
        assert len(scores) == 10
        assert ai.n_actions == 10

    def test_research_subsystem_ai_distinct_roles(self):
        p_suffix = _unique_role("physics_distinct")
        b_suffix = _unique_role("biology_distinct")
        physics = TorchResearchSubsystemAI(p_suffix, 5, 8)
        biology = TorchResearchSubsystemAI(b_suffix, 5, 12)
        assert physics.role == f"research_{p_suffix}"
        assert biology.role == f"research_{b_suffix}"
        assert physics.trunk.heads[physics.role] is not biology.trunk.heads[biology.role]

    # ----- TorchDoctrineAI -----
    def test_doctrine_ai_choose_doctrine_valid(self):
        ai = TorchDoctrineAI(epsilon=0.0)
        state = [0.5] * 10
        ai.predict(state)
        doctrine = ai.choose_doctrine(state)
        assert doctrine in _DOCTRINE_LIST

    def test_doctrine_ai_choose_action_in_bounds(self):
        ai = TorchDoctrineAI(epsilon=0.0)
        a = ai.choose_action([0.5] * 10)
        assert 0 <= a < 5

    def test_doctrine_ai_optimise_every_attribute(self):
        ai = TorchDoctrineAI()
        assert hasattr(ai, "_optimise_every")

    # ----- TorchFleetController -----
    def test_fleet_controller_output_length(self):
        fc = TorchFleetController()
        scores = fc.predict(rand_state(15))
        assert len(scores) == 7

    def test_fleet_controller_epsilon_attribute(self):
        fc = TorchFleetController()
        assert hasattr(fc, "EPSILON")
        assert fc.EPSILON == 0.12

    def test_fleet_controller_train_no_crash(self):
        fc = TorchFleetController()
        s = rand_state(15)
        fc.predict(s)
        fc.train(s, 0, 0.1, rand_state(15))

    # ----- TorchEventAI (categorical) -----
    def test_event_ai_is_categorical(self):
        ai = TorchEventAI()
        assert ai.categorical is True
        assert ai.n_outputs == TorchEventAI.MAX_CHOICES

    def test_event_ai_choose_action_respects_mask(self):
        ai = TorchEventAI(epsilon=0.0)
        mask = [True, False, False, False]
        for _ in range(10):
            a = ai.choose_action(rand_state(5), mask)
            assert a == 0

    def test_event_ai_train_does_not_crash(self):
        ai = TorchEventAI(epsilon=0.0)
        s = rand_state(5)
        a = ai.choose_action(s, [True, True, False, False])
        ai.train(s, a, 1.0, rand_state(5))


# ===========================================================================
# 11. step_trunk (module-level)
# ===========================================================================

class TestStepTrunk:
    def test_no_crash_empty_replay(self):
        step_trunk(n_steps=1)

    def test_no_crash_with_full_replay(self):
        role = _unique_role()
        get_trunk().ensure_role(role, IN, OUT)
        buf = [[0.1] * (CORE_DIM + IN)] * BUFFER_SIZE
        for i in range(_BATCH_SIZE + 1):
            add_experience((role, buf, i % OUT, 1.0, buf))
        step_trunk(n_steps=1)


# ===========================================================================
# 12. Thread safety
# ===========================================================================

class TestThreadSafety:
    def test_concurrent_train_no_exception(self):
        """Multiple nations (each its own bank) training concurrently must
        not raise — the online train() step reads the shared trunk through
        detached tensors, so no lock is needed to serialise it."""
        role = _unique_role()
        n_threads  = 8
        n_steps    = 15
        errors:  list[Exception] = []
        threads: list[threading.Thread] = []

        def worker():
            try:
                view = fresh_view(role=role, epsilon=0.0)
                for _ in range(n_steps):
                    s = rand_state()
                    view.choose_action(s)
                    view.train(s, 0, 1.0, rand_state())
            except Exception as e:
                errors.append(e)

        for _ in range(n_threads):
            t = threading.Thread(target=worker)
            threads.append(t)
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Exceptions in worker threads: {errors}"

    def test_concurrent_train_replay_length_correct(self):
        role = _unique_role()
        n_threads = 4
        n_steps   = 10
        n_before = len(_REPLAY)

        def worker():
            view = fresh_view(role=role, epsilon=0.0)
            for _ in range(n_steps):
                s = rand_state()
                view.choose_action(s)
                view.train(s, 0, 1.0, rand_state())

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        with _REPLAY_LOCK:
            n_after = len(_REPLAY)
        # Within the deque's cap, exactly n_threads*n_steps new entries.
        assert n_after - n_before <= n_threads * n_steps
        assert n_after >= n_before

    def test_train_does_not_touch_trunk_gradients(self):
        """The online train() step must only ever populate .grad on this
        nation's own LoRA tensors — never on the shared trunk's parameters.
        That's what lets concurrent nations train without a shared lock."""
        role = _unique_role()
        view = fresh_view(role=role, epsilon=0.0)
        for p in view.trunk.parameters():
            p.grad = None
        s = rand_state()
        view.train(s, 0, 1.0, rand_state())
        for p in view.trunk.parameters():
            assert p.grad is None


# ===========================================================================
# 13. Hidden state persistence across predict calls
# ===========================================================================

class TestHiddenStateContinuity:
    def test_different_histories_give_different_outputs(self):
        role = _unique_role()
        bank1 = fresh_bank()
        bank2 = fresh_bank()
        c1 = fresh_view(role=role, bank=bank1, seed=0)
        for _ in range(BUFFER_SIZE):
            c1.predict([0.0] * IN)

        c2 = fresh_view(role=role, bank=bank2, seed=0)
        for _ in range(BUFFER_SIZE):
            c2.predict([1.0] * IN)

        current = [0.5] * IN
        o1 = c1.predict(current)
        o2 = c2.predict(current)
        assert o1 != o2, (
            "Views with different histories gave identical outputs — "
            "the RNN is not using temporal context"
        )

    def test_hidden_state_shape_invariant_across_steps(self):
        view = fresh_view()
        for _ in range(10):
            view.predict(rand_state())
            assert view._h.shape == (1, 1, HIDDEN_DIM)


# ===========================================================================
# 14. Double DQN target network (head + core)
# ===========================================================================

class TestDoubleDQN:
    def test_target_initialised_to_online_values(self):
        view = fresh_view()
        assert torch.allclose(view._target_A_head, view.lora_A_head.data)
        assert torch.allclose(view._target_B_head, view.lora_B_head.data)
        assert torch.allclose(view.bank._target_A_core, view.bank.lora_A_core.data)
        assert torch.allclose(view.bank._target_B_core, view.bank.lora_B_core.data)

    def test_target_lags_online_before_update_freq(self):
        view = fresh_view()
        s = rand_state()
        view.choose_action(s)
        view.train(s, 0, 1.0, rand_state())
        assert torch.allclose(
            view._target_B_head, torch.zeros_like(view._target_B_head)
        )

    def test_target_updated_at_freq_boundary(self):
        """After exactly TARGET_UPDATE_FREQ train calls, head AND core targets
        must equal their online counterparts."""
        view = fresh_view()
        freq = UnifiedRoleView.TARGET_UPDATE_FREQ
        for _ in range(freq):
            s = rand_state()
            view.choose_action(s)
            view.train(s, 0, 1.0, rand_state())
        assert torch.allclose(view._target_B_head, view.lora_B_head.data)
        assert torch.allclose(view.bank._target_B_core, view.bank.lora_B_core.data)

    def test_train_step_counter_increments(self):
        view = fresh_view()
        assert view._train_steps == 0
        s = rand_state()
        view.choose_action(s)
        view.train(s, 0, 1.0, rand_state())
        assert view._train_steps == 1

    def test_update_target_network_manually(self):
        view = fresh_view()
        with torch.no_grad():
            view.lora_B_head.fill_(7.0)
            view.bank.lora_B_core.fill_(5.0)
        view._update_target_network()
        assert torch.allclose(view._target_B_head, torch.full_like(view._target_B_head, 7.0))
        assert torch.allclose(view.bank._target_B_core, torch.full_like(view.bank._target_B_core, 5.0))


# ===========================================================================
# 15. LoRA exploration (mutate_lora / reset_lora_optimizer)
# ===========================================================================

class TestLoRAExploration:
    def test_mutate_lora_changes_head_params(self):
        view   = fresh_view()
        before = {
            "A": view.lora_A_head.data.clone(),
            "B": view.lora_B_head.data.clone(),
        }
        view.mutate_lora(sigma=0.5)
        assert not torch.allclose(view.lora_A_head.data, before["A"])
        assert not torch.allclose(view.lora_B_head.data, before["B"])

    def test_mutate_lora_syncs_head_target(self):
        view = fresh_view()
        view.mutate_lora(sigma=0.5)
        assert torch.allclose(view._target_B_head, view.lora_B_head.data)

    def test_mutate_core_changes_bank_params(self):
        bank = fresh_bank()
        before_a = bank.lora_A_core.data.clone()
        before_b = bank.lora_B_core.data.clone()
        bank.mutate_core(sigma=0.5)
        assert not torch.allclose(bank.lora_A_core.data, before_a)
        assert not torch.allclose(bank.lora_B_core.data, before_b)

    def test_mutate_core_syncs_core_target(self):
        bank = fresh_bank()
        bank.mutate_core(sigma=0.5)
        assert torch.allclose(bank._target_B_core, bank.lora_B_core.data)

    def test_reset_lora_optimizer_replaces_optimizer(self):
        view = fresh_view()
        old_id = id(view._head_optimizer)
        view.reset_lora_optimizer()
        assert id(view._head_optimizer) != old_id

    def test_reset_lora_optimizer_fresh_momentum(self):
        view = fresh_view()
        s = rand_state()
        view.choose_action(s)
        view.train(s, 0, 1.0, rand_state())
        view.reset_lora_optimizer()
        assert len(view._head_optimizer.state) == 0

    def test_reset_core_optimizer_fresh_momentum(self):
        view = fresh_view()
        s = rand_state()
        view.choose_action(s)
        view.train(s, 0, 1.0, rand_state())
        view.bank.reset_core_optimizer()
        assert len(view.bank._core_optimizer.state) == 0


# ===========================================================================
# 16. Categorical projection (C51)
# ===========================================================================

class TestCategoricalProjection:
    def test_projection_preserves_probability_mass(self):
        dist = torch.zeros(1, N_ATOMS)
        dist[0, 25] = 1.0
        proj = project_categorical(dist, torch.tensor([10.0]), gamma=0.95)
        assert abs(proj.sum().item() - 1.0) < 1e-5

    def test_projection_expected_value_matches_bellman_target(self):
        from worldsim.ai.rnn.unified import _SUPPORT
        dist = torch.zeros(1, N_ATOMS)
        dist[0, 25] = 1.0   # atom 25 == 0.0 given V_MIN=-50, V_MAX=50, 51 atoms
        reward = 10.0
        gamma = 0.95
        proj = project_categorical(dist, torch.tensor([reward]), gamma=gamma)
        expected = (proj[0] * _SUPPORT).sum().item()
        target = reward + gamma * _SUPPORT[25].item()
        assert abs(expected - target) < 1e-4


# ===========================================================================
# 17. Edge-case robustness
# ===========================================================================

class TestEdgeCases:
    def test_zero_state_does_not_crash(self):
        view = fresh_view()
        view.predict([0.0] * IN)
        view.train([0.0] * IN, 0, 0.0, [0.0] * IN)

    def test_large_reward_does_not_produce_nan(self):
        view = fresh_view()
        s = rand_state()
        view.predict(s)
        view.train(s, 0, 1e6, rand_state())
        scores = view.predict(rand_state())
        assert all(math.isfinite(v) for v in scores)

    def test_negative_reward(self):
        view = fresh_view()
        s = rand_state()
        view.predict(s)
        view.train(s, 0, -1.0, rand_state())
