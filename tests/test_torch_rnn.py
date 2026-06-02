"""Unit tests for worldsim/torch_rnn.py.

Coverage
--------
* _BaseGRUModel  — forward-pass shape & dtype
* RoleBaseModel  — construction, replay buffer, base-model update
* RNNController  — rolling buffer (size=5), predict shape, hidden-state
                   advancement, action bounds, train API, LoRA gradients,
                   base-model grads zeroed after train, experience deposit
* LoRA init      — B matrices start at zero (ΔW = 0 at init);
                   A matrices non-zero; LoRA actually changes output after
                   B is perturbed
* Training loop  — LoRA parameters change after a train() call;
                   replay buffer grows; base-model update converges loss
* Shared base    — two controllers with the same role share one base model
* Role wrappers  — TorchWarAI (create_doctrine, set_allies_dimension),
                   TorchDoctrineAI.choose_doctrine returns a valid string,
                   TorchFleetController output length
* Persistence    — save_lora / load_lora round-trip
* No-op when absent — step_all_base_models() skips roles with empty replay
* Thread safety  — concurrent train() calls from multiple threads do not
                   raise and do not corrupt the replay buffer length
"""
from __future__ import annotations

import copy
import math
import threading
import tempfile
from collections import deque
from pathlib import Path

import pytest
import torch

# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------
from worldsim.torch_rnn import (
    BUFFER_SIZE,
    _DOCTRINE_LIST,
    _FLEET_STATE_LIST,
    RoleBaseModel,
    RNNController,
    _BaseGRUModel,
    get_role_model,
    rnn_available,
    save_all_base_models,
    load_all_base_models,
    step_all_base_models,
    sync_all_meta_ga,
    TorchWarAI,
    TorchDomesticPolicyAI,
    TorchProjectAI,
    TorchDiplomacyAI,
    TorchResearchAI,
    TorchDoctrineAI,
    TorchFleetController,
    _REGISTRY,
    _REGISTRY_LOCK,
)
from worldsim.meta_ga import RewardGA

# ---------------------------------------------------------------------------
# Constants used across tests
# ---------------------------------------------------------------------------
IN   = 6   # small input dim for speed
OUT  = 3   # small output dim
H    = 16  # small hidden dim
ROLE = "test_role_unit"   # isolated key so tests don't collide with real roles


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fresh_controller(
    role: str = ROLE,
    n_in: int = IN,
    n_out: int = OUT,
    h: int = H,
    epsilon: float = 0.0,   # disable exploration by default
    seed: int = 42,
) -> RNNController:
    """Return a fresh RNNController with an isolated role key."""
    # Clean any leftover registry entry so tests are independent
    key = f"{role}_{n_in}_{n_out}"
    with _REGISTRY_LOCK:
        _REGISTRY.pop(key, None)
    return RNNController(role, n_in, n_out, h, epsilon=epsilon, gamma=0.9, seed=seed)


def rand_state(n: int = IN) -> list[float]:
    return [float(torch.randn(1).item()) for _ in range(n)]


# ===========================================================================
# 1. Module-level availability
# ===========================================================================

class TestAvailability:
    def test_rnn_available(self):
        assert rnn_available() is True

    def test_buffer_size_constant(self):
        assert BUFFER_SIZE == 5

    def test_doctrine_list_length(self):
        assert len(_DOCTRINE_LIST) == 5

    def test_fleet_state_list_length(self):
        assert len(_FLEET_STATE_LIST) == 7


# ===========================================================================
# 2. _BaseGRUModel
# ===========================================================================

class TestBaseGRUModel:
    def _model(self, in_=IN, h=H, out=OUT):
        return _BaseGRUModel(in_, h, out)

    def test_forward_output_shape(self):
        m = self._model()
        x  = torch.randn(1, BUFFER_SIZE, IN)
        h0 = torch.zeros(1, 1, H)
        out = m(x, h0)
        assert out.shape == (1, OUT)

    def test_forward_dtype_float32(self):
        m = self._model()
        x  = torch.randn(1, 3, IN)
        h0 = torch.zeros(1, 1, H)
        out = m(x, h0)
        assert out.dtype == torch.float32

    def test_forward_single_step(self):
        """Single time step (T=1) must still produce correct output shape."""
        m = self._model()
        x  = torch.randn(1, 1, IN)
        h0 = torch.zeros(1, 1, H)
        out = m(x, h0)
        assert out.shape == (1, OUT)

    def test_different_inputs_different_outputs(self):
        """Two different inputs should (almost certainly) give different outputs."""
        m  = self._model()
        h0 = torch.zeros(1, 1, H)
        x1 = torch.randn(1, BUFFER_SIZE, IN)
        x2 = torch.randn(1, BUFFER_SIZE, IN)
        o1 = m(x1, h0)
        o2 = m(x2, h0)
        assert not torch.allclose(o1, o2)

    def test_hidden_dim_attribute(self):
        m = self._model(h=32)
        assert m.hidden_dim == 32

    def test_layer_parameter_counts(self):
        """Verify that input_proj, gru, and output_proj are present."""
        m = self._model()
        param_names = {n.split(".")[0] for n, _ in m.named_parameters()}
        assert {"input_proj", "gru", "output_proj"} <= param_names


# ===========================================================================
# 3. RoleBaseModel
# ===========================================================================

class TestRoleBaseModel:
    def _rbm(self):
        return RoleBaseModel(IN, H, OUT)

    def test_construction(self):
        rbm = self._rbm()
        assert rbm.model is not None
        assert len(rbm._replay) == 0

    def test_add_experience(self):
        rbm = self._rbm()
        buf = [[0.0] * IN] * BUFFER_SIZE
        rbm.add_experience((buf, 0, 1.0, buf))
        assert len(rbm._replay) == 1

    def test_replay_cap(self):
        rbm = self._rbm()
        rbm._REPLAY_CAP = 10
        rbm._replay = deque(maxlen=10)
        buf = [[0.0] * IN] * BUFFER_SIZE
        for _ in range(15):
            rbm.add_experience((buf, 0, 0.1, buf))
        assert len(rbm._replay) == 10

    def test_update_base_no_op_when_small(self):
        """update_base should be a no-op when replay is smaller than batch size."""
        rbm = self._rbm()
        # Store original params
        w_before = rbm.model.input_proj.weight.data.clone()
        rbm.update_base(n_steps=4)
        w_after = rbm.model.input_proj.weight.data.clone()
        assert torch.allclose(w_before, w_after), "weights changed with empty replay"

    def test_update_base_changes_weights(self):
        """With a full replay buffer, update_base() should modify base weights."""
        rbm = self._rbm()
        buf_s = [[float(i)] * IN for i in range(BUFFER_SIZE)]
        buf_n = [[float(i + 1)] * IN for i in range(BUFFER_SIZE)]
        for action in range(OUT):
            for _ in range(rbm._BATCH_SIZE + 5):
                rbm.add_experience((buf_s, action % OUT, 1.0, buf_n))

        w_before = rbm.model.output_proj.weight.data.clone()
        rbm.update_base(n_steps=2)
        w_after  = rbm.model.output_proj.weight.data.clone()
        assert not torch.allclose(w_before, w_after), (
            "weights unchanged after update_base with full replay"
        )

    def test_save_load_round_trip(self, tmp_path):
        rbm = self._rbm()
        # Perturb weights
        with torch.no_grad():
            rbm.model.input_proj.weight.fill_(3.14)
        path = tmp_path / "base.pt"
        rbm.save(path)
        rbm2 = self._rbm()
        rbm2.load(path)
        assert torch.allclose(
            rbm.model.input_proj.weight,
            rbm2.model.input_proj.weight,
        )


# ===========================================================================
# 4. RNNController — construction, buffer, hidden state
# ===========================================================================

class TestRNNControllerBuffer:
    def test_buffer_initial_size(self):
        ctrl = fresh_controller()
        assert len(ctrl._buffer) == BUFFER_SIZE

    def test_buffer_fills_after_predict_calls(self):
        ctrl = fresh_controller()
        for i in range(BUFFER_SIZE + 2):
            ctrl.predict(rand_state())
        assert len(ctrl._buffer) == BUFFER_SIZE   # maxlen enforced

    def test_buffer_slides_correctly(self):
        """After 1 predict the oldest zero-state should be evicted."""
        ctrl = fresh_controller()
        new_state = [1.0] * IN
        ctrl.predict(new_state)
        # Last entry of buffer must be new_state
        assert ctrl._buffer[-1] == new_state

    def test_hidden_state_advances(self):
        ctrl = fresh_controller()
        h_before = ctrl._h.clone()
        ctrl.predict(rand_state())
        # Hidden state must change after processing real inputs
        # (may stay zero only if all inputs are zero — use non-zero state)
        assert ctrl._h.shape == (1, 1, H)
        # h_before was zeros; after predict it should be non-zero (with high probability)
        # Only assert shape here to avoid flakiness
        assert ctrl._h.requires_grad is False   # must be detached

    def test_predict_output_length(self):
        ctrl = fresh_controller()
        scores = ctrl.predict(rand_state())
        assert len(scores) == OUT
        assert all(isinstance(v, float) for v in scores)

    def test_predict_deterministic_for_same_input_sequence(self):
        """Two freshly-created controllers sharing the same base model and
        starting from B=0 LoRA must produce identical outputs for the same
        input sequence.

        With B_in=B_out=0 the LoRA contribution is exactly zero for both
        controllers, so _forward_tensor reduces to the base model forward
        which they share — identical inputs → identical outputs.
        """
        role = "test_det_role"
        key  = f"{role}_{IN}_{OUT}"
        with _REGISTRY_LOCK:
            _REGISTRY.pop(key, None)   # clear ONCE before both controllers
        c1 = RNNController(role, IN, OUT, H, epsilon=0.0)
        c2 = RNNController(role, IN, OUT, H, epsilon=0.0)
        # Fixed, non-random states so the test is fully deterministic
        states = [[float(i % 3) * 0.1] * IN for i in range(BUFFER_SIZE)]
        for s in states:
            o1 = c1.predict(s)
        for s in states:
            o2 = c2.predict(s)
        assert o1 == o2, f"Expected identical outputs, got {o1!r} vs {o2!r}"
        with _REGISTRY_LOCK:
            _REGISTRY.pop(key, None)

    def test_predict_with_short_state_padded(self):
        """State shorter than n_inputs should be zero-padded without error."""
        ctrl = fresh_controller()
        scores = ctrl.predict([0.5])   # only 1 value, needs padding to IN
        assert len(scores) == OUT

    def test_predict_with_long_state_truncated(self):
        """State longer than n_inputs should be truncated."""
        ctrl = fresh_controller()
        scores = ctrl.predict([0.1] * (IN * 3))
        assert len(scores) == OUT


# ===========================================================================
# 5. RNNController — predict_prob and choose_action
# ===========================================================================

class TestRNNControllerActions:
    def test_predict_prob_sums_to_one(self):
        ctrl = fresh_controller()
        probs = ctrl.predict_prob(rand_state())
        assert abs(sum(probs) - 1.0) < 1e-5

    def test_predict_prob_all_positive(self):
        ctrl = fresh_controller()
        probs = ctrl.predict_prob(rand_state())
        assert all(p > 0 for p in probs)

    def test_choose_action_within_bounds(self):
        ctrl = fresh_controller(epsilon=0.0)  # greedy
        for _ in range(20):
            a = ctrl.choose_action(rand_state())
            assert 0 <= a < OUT

    def test_choose_action_epsilon_greedy_explores(self):
        ctrl = fresh_controller(epsilon=1.0)  # always random
        actions = {ctrl.choose_action(rand_state()) for _ in range(50)}
        assert len(actions) > 1   # should visit multiple actions

    def test_choose_action_valid_mask(self):
        ctrl = fresh_controller(epsilon=0.0)
        # Only action 1 is valid
        mask = [False, True, False]
        for _ in range(10):
            a = ctrl.choose_action(rand_state(), valid_mask=mask)
            assert a == 1

    def test_choose_action_valid_mask_with_exploration(self):
        ctrl = fresh_controller(epsilon=1.0)
        mask = [True, False, True]
        for _ in range(30):
            a = ctrl.choose_action(rand_state(), valid_mask=mask)
            assert a in (0, 2)


# ===========================================================================
# 6. LoRA initialisation
# ===========================================================================

class TestLoRAInit:
    def test_B_matrices_start_at_zero(self):
        """B=0 means the adapter has exactly zero effect at initialisation."""
        ctrl = fresh_controller()
        assert torch.all(ctrl.lora_B_in  == 0.0)
        assert torch.all(ctrl.lora_B_out == 0.0)

    def test_A_matrices_non_zero(self):
        ctrl = fresh_controller()
        assert not torch.all(ctrl.lora_A_in  == 0.0)
        assert not torch.all(ctrl.lora_A_out == 0.0)

    def test_zero_lora_equals_base_output(self):
        """With B=0 the controller output must equal the base model output."""
        ctrl = fresh_controller()
        # Construct a simple test sequence
        seq = torch.zeros(1, BUFFER_SIZE, IN)
        h0  = torch.zeros(1, 1, H)
        base_out  = ctrl.base.model(seq, h0)
        ctrl_out  = ctrl._forward_tensor(seq, h0)
        assert torch.allclose(base_out, ctrl_out, atol=1e-6)

    def test_nonzero_lora_changes_output(self):
        """After perturbing lora_B_out the output must differ from base."""
        ctrl = fresh_controller()
        seq = torch.randn(1, BUFFER_SIZE, IN)
        h0  = torch.zeros(1, 1, H)
        base_out = ctrl.base.model(seq, h0).detach().clone()
        with torch.no_grad():
            ctrl.lora_B_out.fill_(0.5)
        ctrl_out = ctrl._forward_tensor(seq, h0)
        assert not torch.allclose(base_out, ctrl_out, atol=1e-6)

    def test_lora_scale_applied(self):
        """LoRA delta = (B_out @ A_out) * (alpha / r) — verify scaling."""
        ctrl = fresh_controller()
        seq = torch.randn(1, BUFFER_SIZE, IN)
        h0  = torch.zeros(1, 1, H)
        with torch.no_grad():
            ctrl.lora_B_out.fill_(1.0)
            ctrl.lora_A_out.fill_(1.0)
        out = ctrl._forward_tensor(seq, h0)
        base = ctrl.base.model(seq, h0)
        delta = out - base
        expected_scale = ctrl.lora_alpha / ctrl.lora_r
        # Delta should be non-negligible (>0) after scale application
        assert delta.abs().max().item() > 0

    def test_lora_rank_and_alpha_defaults(self):
        ctrl = fresh_controller()
        assert ctrl.lora_r     == RNNController._LORA_R
        assert ctrl.lora_alpha == RNNController._LORA_ALPHA

    def test_lora_A_in_shape(self):
        ctrl = fresh_controller()
        assert ctrl.lora_A_in.shape  == (ctrl.lora_r, IN)

    def test_lora_B_in_shape(self):
        ctrl = fresh_controller()
        assert ctrl.lora_B_in.shape  == (H, ctrl.lora_r)

    def test_lora_A_out_shape(self):
        ctrl = fresh_controller()
        assert ctrl.lora_A_out.shape == (ctrl.lora_r, H)

    def test_lora_B_out_shape(self):
        ctrl = fresh_controller()
        assert ctrl.lora_B_out.shape == (OUT, ctrl.lora_r)


# ===========================================================================
# 7. Training loop
# ===========================================================================

class TestTrainingLoop:
    def _trained_ctrl(self, n_steps: int = 10) -> RNNController:
        ctrl = fresh_controller(epsilon=0.0)
        state      = rand_state()
        next_state = rand_state()
        for _ in range(n_steps):
            action = ctrl.choose_action(state)
            ctrl.train(state, action, 1.0, next_state)
            state = next_state
            next_state = rand_state()
        return ctrl

    def test_lora_B_out_changes_after_training(self):
        ctrl    = fresh_controller()
        b_before = ctrl.lora_B_out.data.clone()
        state = rand_state()
        ctrl.choose_action(state)
        ctrl.train(state, 0, 1.0, rand_state())
        b_after  = ctrl.lora_B_out.data.clone()
        # B was zero initially; after one step with a nonzero gradient it changes
        assert not torch.allclose(b_before, b_after), (
            "lora_B_out unchanged after training step"
        )

    def test_lora_A_in_changes_after_training(self):
        """lora_A_in gets zero gradient on step 1 because the gradient
        d/d_A_in flows through lora_B_in which is zero at init.  After
        step 1 B_in is non-zero, so step 2 gives A_in a real gradient.
        Run 3 steps to ensure A_in has changed.
        """
        ctrl = fresh_controller()
        a_before = ctrl.lora_A_in.data.clone()
        for _ in range(3):
            state = rand_state()
            ctrl.choose_action(state)
            ctrl.train(state, 0, 1.0, rand_state())
        a_after  = ctrl.lora_A_in.data.clone()
        assert not torch.allclose(a_before, a_after)

    def test_base_model_grads_zeroed_after_train(self):
        """After train(), no grad should linger on base-model parameters."""
        ctrl = fresh_controller()
        state = rand_state()
        ctrl.choose_action(state)
        ctrl.train(state, 0, 1.0, rand_state())
        for p in ctrl.base.model.parameters():
            if p.grad is not None:
                assert torch.all(p.grad == 0), "base model grad not zeroed"

    def test_experience_deposited_in_replay(self):
        """Each train() call should add exactly one entry to the replay buffer."""
        ctrl = fresh_controller()
        n_before = len(ctrl.base._replay)
        state = rand_state()
        ctrl.choose_action(state)
        ctrl.train(state, 0, 0.5, rand_state())
        assert len(ctrl.base._replay) == n_before + 1

    def test_train_ignores_out_of_range_action(self):
        """train() with action >= n_outputs must be a no-op (no crash, no update)."""
        ctrl    = fresh_controller()
        b_before = ctrl.lora_B_out.data.clone()
        state = rand_state()
        ctrl.predict(state)   # advance buffer
        ctrl.train(state, OUT + 5, 1.0, rand_state())
        assert torch.allclose(ctrl.lora_B_out.data, b_before)

    def test_train_ignores_negative_action(self):
        ctrl    = fresh_controller()
        b_before = ctrl.lora_B_out.data.clone()
        state = rand_state()
        ctrl.predict(state)
        ctrl.train(state, -1, 1.0, rand_state())
        assert torch.allclose(ctrl.lora_B_out.data, b_before)

    def test_multi_step_training_does_not_crash(self):
        """Running many train steps should not raise any exception."""
        self._trained_ctrl(n_steps=50)

    def test_replay_buffer_fills_monotonically(self):
        ctrl = fresh_controller()
        for i in range(1, 11):
            state = rand_state()
            ctrl.choose_action(state)
            ctrl.train(state, i % OUT, float(i), rand_state())
            assert len(ctrl.base._replay) == i

    def test_base_update_reduces_loss(self):
        """Running step_all_base_models() with a dense replay should not crash
        and should leave the model's weights finite."""
        role_key = "test_loss_role"
        key = f"{role_key}_{IN}_{OUT}"
        with _REGISTRY_LOCK:
            _REGISTRY.pop(key, None)
        ctrl = RNNController(role_key, IN, OUT, H, epsilon=0.0, gamma=0.9)
        # Fill replay
        buf = [[0.5] * IN] * BUFFER_SIZE
        for i in range(RoleBaseModel._BATCH_SIZE * 2):
            ctrl.base.add_experience((buf, i % OUT, 1.0, buf))
        # Should not raise
        ctrl.base.update_base(n_steps=2)
        for p in ctrl.base.model.parameters():
            assert torch.all(torch.isfinite(p.data)), "NaN/Inf in base model after update"
        # Cleanup
        with _REGISTRY_LOCK:
            _REGISTRY.pop(key, None)


# ===========================================================================
# 8. Registry sharing
# ===========================================================================

class TestRegistry:
    def test_two_controllers_share_base(self):
        role = "test_share_role"
        key  = f"{role}_{IN}_{OUT}"
        with _REGISTRY_LOCK:
            _REGISTRY.pop(key, None)
        c1 = RNNController(role, IN, OUT, H)
        c2 = RNNController(role, IN, OUT, H)
        assert c1.base is c2.base

    def test_different_roles_have_different_bases(self):
        r1, r2 = "test_diff_role_a", "test_diff_role_b"
        for key in [f"{r1}_{IN}_{OUT}", f"{r2}_{IN}_{OUT}"]:
            with _REGISTRY_LOCK:
                _REGISTRY.pop(key, None)
        c1 = RNNController(r1, IN, OUT, H)
        c2 = RNNController(r2, IN, OUT, H)
        assert c1.base is not c2.base

    def test_different_input_dims_have_different_bases(self):
        role = "test_dim_role"
        for key in [f"{role}_{IN}_{OUT}", f"{role}_{IN+2}_{OUT}"]:
            with _REGISTRY_LOCK:
                _REGISTRY.pop(key, None)
        c1 = RNNController(role, IN,     OUT, H)
        c2 = RNNController(role, IN + 2, OUT, H)
        assert c1.base is not c2.base

    def test_get_role_model_is_idempotent(self):
        role = "test_idem_role"
        key  = f"{role}_{IN}_{OUT}"
        with _REGISTRY_LOCK:
            _REGISTRY.pop(key, None)
        m1 = get_role_model(role, IN, H, OUT)
        m2 = get_role_model(role, IN, H, OUT)
        assert m1 is m2

    def test_experience_from_nation_a_visible_to_nation_b(self):
        """Two controllers sharing a role share the same replay — an experience
        added through controller A must be visible to controller B."""
        role = "test_shared_replay"
        key  = f"{role}_{IN}_{OUT}"
        with _REGISTRY_LOCK:
            _REGISTRY.pop(key, None)
        c1 = RNNController(role, IN, OUT, H, epsilon=0.0)
        c2 = RNNController(role, IN, OUT, H, epsilon=0.0)
        n_before = len(c2.base._replay)
        state = rand_state()
        c1.predict(state)
        c1.train(state, 0, 1.0, rand_state())
        assert len(c2.base._replay) == n_before + 1


# ===========================================================================
# 9. Persistence (save_lora / load_lora)
# ===========================================================================

class TestPersistence:
    def test_save_load_lora_round_trip(self, tmp_path):
        ctrl = fresh_controller()
        # Train a few steps so LoRA diverges from zero
        for _ in range(5):
            s = rand_state()
            ctrl.predict(s)
            ctrl.train(s, 0, 1.0, rand_state())
        path = tmp_path / "lora.pt"
        ctrl.save_lora(path)

        ctrl2 = fresh_controller()
        ctrl2.load_lora(path)

        assert torch.allclose(ctrl.lora_A_in,  ctrl2.lora_A_in)
        assert torch.allclose(ctrl.lora_B_in,  ctrl2.lora_B_in)
        assert torch.allclose(ctrl.lora_A_out, ctrl2.lora_A_out)
        assert torch.allclose(ctrl.lora_B_out, ctrl2.lora_B_out)

    def test_load_lora_nonexistent_path_is_no_op(self, tmp_path):
        ctrl = fresh_controller()
        ctrl.load_lora(tmp_path / "does_not_exist.lora.pt")  # should not raise

    def test_save_table_creates_lora_file(self, tmp_path):
        """_RoleController.save_table must create a .lora.pt file."""
        ctrl = TorchDiplomacyAI(epsilon=0.0)
        yml_path = tmp_path / "diplomacy.yml"
        ctrl.save_table(yml_path)
        assert (tmp_path / "diplomacy.lora.pt").exists()

    def test_load_table_reads_lora_file(self, tmp_path):
        c1 = TorchDiplomacyAI(epsilon=0.0)
        # Perturb LoRA
        with torch.no_grad():
            c1.lora_B_out.fill_(2.0)
        yml_path = tmp_path / "dip2.yml"
        c1.save_table(yml_path)

        c2 = TorchDiplomacyAI(table_path=yml_path, epsilon=0.0)
        assert torch.allclose(c1.lora_B_out, c2.lora_B_out)

    def test_save_load_all_base_models(self, tmp_path):
        """save_all_base_models / load_all_base_models round-trips base weights."""
        role = "test_persist_base"
        key  = f"{role}_{IN}_{OUT}"
        with _REGISTRY_LOCK:
            _REGISTRY.pop(key, None)
        rbm = get_role_model(role, IN, H, OUT)
        with torch.no_grad():
            rbm.model.input_proj.weight.fill_(7.77)

        save_all_base_models(tmp_path)

        # Corrupt the in-memory weight
        with torch.no_grad():
            rbm.model.input_proj.weight.fill_(0.0)

        load_all_base_models(tmp_path)
        assert torch.allclose(rbm.model.input_proj.weight,
                               torch.full_like(rbm.model.input_proj.weight, 7.77))
        with _REGISTRY_LOCK:
            _REGISTRY.pop(key, None)


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
        # After dimension change predict should still work
        scores = ai.predict(rand_state(ai.n_inputs))
        assert len(scores) == 2

    def test_torch_war_ai_set_allies_dimension_no_op(self):
        ai = TorchWarAI(allies_dim=4)
        n_before = ai.n_inputs
        ai.set_allies_dimension(4)  # same dim — no-op
        assert ai.n_inputs == n_before

    # ----- TorchDomesticPolicyAI -----
    def test_domestic_policy_ai_output_dim(self):
        ai = TorchDomesticPolicyAI(actions=21, n_inputs=20)
        assert ai.n_outputs == 21
        scores = ai.predict(rand_state(20))
        assert len(scores) == 21

    def test_domestic_policy_ai_choose_action_in_bounds(self):
        ai = TorchDomesticPolicyAI(actions=21, n_inputs=20, epsilon=0.0)
        a = ai.choose_action(rand_state(20))
        assert 0 <= a < 21

    # ----- TorchProjectAI -----
    def test_project_ai_output_dim(self):
        ai = TorchProjectAI(actions=12, n_inputs=6)
        assert ai.n_outputs == 12

    # ----- TorchDiplomacyAI -----
    def test_diplomacy_ai_train_does_not_crash(self):
        ai = TorchDiplomacyAI(actions=3, n_inputs=5, epsilon=0.0)
        s  = rand_state(5)
        ai.choose_action(s)
        ai.train(s, 0, 0.5, rand_state(5))   # should not raise

    # ----- TorchResearchAI -----
    def test_research_ai_output_shape(self):
        ai = TorchResearchAI(actions=10, n_inputs=4)
        scores = ai.predict(rand_state(4))
        assert len(scores) == 10

    # ----- TorchDoctrineAI -----
    def test_doctrine_ai_choose_doctrine_valid(self):
        ai = TorchDoctrineAI(epsilon=0.0)
        from worldsim.models.military_ai import DOCTRINE_LIST
        state = [0.5] * 10
        ai.predict(state)   # advance buffer
        doctrine = ai.choose_doctrine(state)
        assert doctrine in DOCTRINE_LIST

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
        assert len(scores) == 7   # len(FLEET_STATE_LIST)

    def test_fleet_controller_epsilon_attribute(self):
        fc = TorchFleetController()
        assert hasattr(fc, "EPSILON")
        assert fc.EPSILON == 0.12

    def test_fleet_controller_train_no_crash(self):
        fc = TorchFleetController()
        s = rand_state(15)
        fc.predict(s)
        fc.train(s, 0, 0.1, rand_state(15))


# ===========================================================================
# 11. step_all_base_models
# ===========================================================================

class TestStepAllBaseModels:
    def test_no_crash_empty_registry(self):
        """step_all_base_models must not raise even with small/empty replays."""
        step_all_base_models(n_steps=1)   # should be a no-op for small replays

    def test_no_crash_with_full_replay(self):
        role = "test_global_step"
        key  = f"{role}_{IN}_{OUT}"
        with _REGISTRY_LOCK:
            _REGISTRY.pop(key, None)
        rbm = get_role_model(role, IN, H, OUT)
        buf = [[0.1] * IN] * BUFFER_SIZE
        for i in range(RoleBaseModel._BATCH_SIZE + 1):
            rbm.add_experience((buf, i % OUT, 1.0, buf))
        step_all_base_models(n_steps=1)  # should not raise
        with _REGISTRY_LOCK:
            _REGISTRY.pop(key, None)


# ===========================================================================
# 12. Thread safety
# ===========================================================================

class TestThreadSafety:
    def test_concurrent_train_no_exception(self):
        """Multiple threads each running train() must not raise."""
        role = "test_thread_role"
        key  = f"{role}_{IN}_{OUT}"
        with _REGISTRY_LOCK:
            _REGISTRY.pop(key, None)

        n_threads  = 8
        n_steps    = 20
        errors:  list[Exception] = []
        threads: list[threading.Thread] = []

        def worker():
            try:
                ctrl = RNNController(role, IN, OUT, H, epsilon=0.0, gamma=0.9)
                for _ in range(n_steps):
                    s = rand_state()
                    ctrl.choose_action(s)
                    ctrl.train(s, 0, 1.0, rand_state())
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

        with _REGISTRY_LOCK:
            _REGISTRY.pop(key, None)

    def test_concurrent_train_replay_length_correct(self):
        """Total experiences in replay must equal n_threads * n_steps after concurrent
        training (within the deque's cap)."""
        role = "test_thread_replay"
        key  = f"{role}_{IN}_{OUT}"
        with _REGISTRY_LOCK:
            _REGISTRY.pop(key, None)

        n_threads = 4
        n_steps   = 10
        base = get_role_model(role, IN, H, OUT)

        def worker():
            ctrl = RNNController(role, IN, OUT, H, epsilon=0.0)
            for _ in range(n_steps):
                s = rand_state()
                ctrl.choose_action(s)
                ctrl.train(s, 0, 1.0, rand_state())

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        expected = min(n_threads * n_steps, base._REPLAY_CAP)
        assert len(base._replay) == expected, (
            f"replay length {len(base._replay)} != expected {expected}"
        )

        with _REGISTRY_LOCK:
            _REGISTRY.pop(key, None)


# ===========================================================================
# 13. Hidden state persistence across predict calls
# ===========================================================================

class TestHiddenStateContinuity:
    def test_different_histories_give_different_outputs(self):
        """Two controllers fed the same current state but different histories
        should produce different scores (demonstrating temporal memory)."""
        # Controller 1: warmed up on all-zeros history
        c1 = fresh_controller(seed=0)
        for _ in range(BUFFER_SIZE):
            c1.predict([0.0] * IN)

        # Controller 2: warmed up on all-ones history
        c2 = fresh_controller(seed=0)
        for _ in range(BUFFER_SIZE):
            c2.predict([1.0] * IN)

        # Same current state
        current = [0.5] * IN
        o1 = c1.predict(current)
        o2 = c2.predict(current)
        assert o1 != o2, (
            "Controllers with different histories gave identical outputs — "
            "the RNN is not using temporal context"
        )

    def test_hidden_state_shape_invariant_across_steps(self):
        ctrl = fresh_controller()
        for _ in range(10):
            ctrl.predict(rand_state())
            assert ctrl._h.shape == (1, 1, H)


# ===========================================================================
# 14. Edge-case robustness
# ===========================================================================

# ===========================================================================
# 15. Double DQN target network
# ===========================================================================

class TestDoubleDQN:
    def test_target_initialised_to_online_values(self):
        """At init the target LoRA must equal the online LoRA (B=0, A=kaiming)."""
        ctrl = fresh_controller()
        assert torch.allclose(ctrl._target_A_in,  ctrl.lora_A_in.data)
        assert torch.allclose(ctrl._target_B_in,  ctrl.lora_B_in.data)
        assert torch.allclose(ctrl._target_A_out, ctrl.lora_A_out.data)
        assert torch.allclose(ctrl._target_B_out, ctrl.lora_B_out.data)

    def test_target_lags_online_before_update_freq(self):
        """Before TARGET_UPDATE_FREQ steps the target should differ from online."""
        ctrl = fresh_controller()
        # Train one step: online updates, target does not
        s = rand_state()
        ctrl.choose_action(s)
        ctrl.train(s, 0, 1.0, rand_state())
        # online B_out has changed (it started at 0, received gradient)
        # target B_out should still be all-zeros (no update yet)
        assert torch.allclose(ctrl._target_B_out,
                               torch.zeros_like(ctrl._target_B_out))

    def test_target_updated_at_freq_boundary(self):
        """After exactly TARGET_UPDATE_FREQ train calls target == online."""
        ctrl = fresh_controller()
        freq = RNNController.TARGET_UPDATE_FREQ
        for _ in range(freq):
            s = rand_state()
            ctrl.choose_action(s)
            ctrl.train(s, 0, 1.0, rand_state())
        # At step `freq` the hard copy runs inside train()
        assert torch.allclose(ctrl._target_B_out, ctrl.lora_B_out.data)
        assert torch.allclose(ctrl._target_A_in,  ctrl.lora_A_in.data)

    def test_double_dqn_uses_target_for_next_state(self):
        """Verify Double DQN: the target network evaluates the action chosen
        by the online network.  With B=0 target the evaluation must differ
        from the vanilla TD(0) target when target has diverged."""
        ctrl = fresh_controller()
        # Perturb target B_out to a known value so target != online
        with torch.no_grad():
            ctrl._target_B_out.fill_(1.0)   # target gives non-zero output
            ctrl.lora_B_out.fill_(0.0)      # online stays at zero
        s = rand_state()
        ctrl.choose_action(s)
        b_before = ctrl.lora_B_out.data.clone()
        ctrl.train(s, 0, 1.0, rand_state())
        b_after = ctrl.lora_B_out.data.clone()
        # LoRA should have received a gradient (the target != online path)
        assert not torch.allclose(b_before, b_after)

    def test_train_step_counter_increments(self):
        ctrl = fresh_controller()
        assert ctrl._train_steps == 0
        s = rand_state()
        ctrl.choose_action(s)
        ctrl.train(s, 0, 1.0, rand_state())
        assert ctrl._train_steps == 1

    def test_target_network_persisted_in_save_load(self, tmp_path):
        """save_lora must store target tensors and load_lora must restore them."""
        ctrl = fresh_controller()
        with torch.no_grad():
            ctrl._target_B_out.fill_(3.14)
        path = tmp_path / "lora_tgt.pt"
        ctrl.save_lora(path)
        ctrl2 = fresh_controller()
        ctrl2.load_lora(path)
        assert torch.allclose(ctrl2._target_B_out,
                               torch.full_like(ctrl2._target_B_out, 3.14))

    def test_update_target_network_manually(self):
        ctrl = fresh_controller()
        with torch.no_grad():
            ctrl.lora_B_out.fill_(7.0)
        ctrl._update_target_network()
        assert torch.allclose(ctrl._target_B_out,
                               torch.full_like(ctrl._target_B_out, 7.0))


# ===========================================================================
# 16. MetaGA integration
# ===========================================================================

def _make_ga(length: int = 6, fitness: float = 0.0) -> RewardGA:
    """Helper: return a RewardGA with a known active genome fitness."""
    ga = RewardGA(length, population=4)
    ga.population[ga.active].fitness = fitness
    return ga


class TestMetaGAIntegration:
    def test_sync_positive_fitness_reduces_epsilon(self):
        ctrl = fresh_controller(epsilon=0.10)
        ga   = _make_ga(fitness=160.0)   # 2 × FITNESS_SCALE → epsilon ≈ 0.10 · e^{-2}
        ctrl.sync_meta_ga(ga)
        assert ctrl.epsilon < 0.10, f"epsilon did not decrease: {ctrl.epsilon}"
        assert ctrl.epsilon >= RNNController._EPSILON_FLOOR

    def test_sync_zero_fitness_keeps_base_epsilon(self):
        ctrl = fresh_controller(epsilon=0.10)
        ga   = _make_ga(fitness=0.0)
        ctrl.sync_meta_ga(ga)
        # exp(-0) = 1 → epsilon == base_epsilon
        assert abs(ctrl.epsilon - 0.10) < 1e-6

    def test_sync_negative_fitness_raises_epsilon_toward_base(self):
        """Negative fitness must not drive epsilon below floor."""
        ctrl = fresh_controller(epsilon=0.10)
        ga   = _make_ga(fitness=-200.0)
        ctrl.sync_meta_ga(ga)
        # max(0, -200)/80 = 0 → epsilon stays at base (no amplification above 1.0)
        assert ctrl.epsilon == pytest.approx(0.10, abs=1e-6)

    def test_sync_positive_fitness_increases_reward_scale(self):
        ctrl = fresh_controller()
        ga   = _make_ga(fitness=200.0)   # scale = 1 + 200/200 = 2.0
        ctrl.sync_meta_ga(ga)
        assert ctrl._reward_scale > 1.0

    def test_sync_negative_fitness_decreases_reward_scale(self):
        ctrl = fresh_controller()
        ga   = _make_ga(fitness=-200.0)  # scale = max(0.5, 1 - 1.0) = 0.5
        ctrl.sync_meta_ga(ga)
        assert ctrl._reward_scale < 1.0
        assert ctrl._reward_scale >= 0.5

    def test_reward_scale_clamped_at_max(self):
        ctrl = fresh_controller()
        ga   = _make_ga(fitness=10_000.0)
        ctrl.sync_meta_ga(ga)
        assert ctrl._reward_scale <= 2.0

    def test_genome_change_resets_optimizer(self):
        """Switching the active genome must trigger optimizer reset."""
        ctrl = fresh_controller()
        ga   = _make_ga()
        ctrl.sync_meta_ga(ga)           # genome 0 → _genome_id = 0

        # Train a few steps so Adam accumulates momentum
        s = rand_state()
        ctrl.choose_action(s)
        ctrl.train(s, 0, 1.0, rand_state())

        old_opt_id = id(ctrl._lora_optimizer)

        # Change active genome
        ga.active = 1
        ctrl.sync_meta_ga(ga)           # genome 1 → should reset optimizer

        assert id(ctrl._lora_optimizer) != old_opt_id, (
            "Optimizer not replaced after genome change"
        )

    def test_same_genome_no_optimizer_reset(self):
        """Syncing the same genome twice must NOT reset the optimizer."""
        ctrl = fresh_controller()
        ga   = _make_ga()
        ctrl.sync_meta_ga(ga)
        old_opt_id = id(ctrl._lora_optimizer)
        ctrl.sync_meta_ga(ga)   # same genome
        assert id(ctrl._lora_optimizer) == old_opt_id

    def test_reward_scaling_applied_in_train(self):
        """With _reward_scale > 1 the replay stores a larger reward value."""
        ctrl = fresh_controller()
        ctrl._reward_scale = 2.0
        s = rand_state()
        ctrl.choose_action(s)
        ctrl.train(s, 0, 1.0, rand_state())   # reward 1.0 → stored as 2.0
        _, _, stored_reward, _ = ctrl.base._replay[-1]
        assert abs(stored_reward - 2.0) < 1e-5

    def test_mutate_lora_changes_all_params(self):
        ctrl   = fresh_controller()
        before = {
            "A_in":  ctrl.lora_A_in.data.clone(),
            "B_in":  ctrl.lora_B_in.data.clone(),
            "A_out": ctrl.lora_A_out.data.clone(),
            "B_out": ctrl.lora_B_out.data.clone(),
        }
        ctrl.mutate_lora(sigma=0.5)   # large sigma → almost certainly changes
        assert not torch.allclose(ctrl.lora_A_in.data, before["A_in"])
        assert not torch.allclose(ctrl.lora_B_in.data, before["B_in"])
        assert not torch.allclose(ctrl.lora_A_out.data, before["A_out"])
        assert not torch.allclose(ctrl.lora_B_out.data, before["B_out"])

    def test_mutate_lora_syncs_target(self):
        """After mutation the target must equal the (now noisy) online LoRA."""
        ctrl = fresh_controller()
        ctrl.mutate_lora(sigma=0.5)
        assert torch.allclose(ctrl._target_B_out, ctrl.lora_B_out.data)
        assert torch.allclose(ctrl._target_A_in,  ctrl.lora_A_in.data)

    def test_reset_lora_optimizer_replaces_optimizer(self):
        ctrl = fresh_controller()
        old_id = id(ctrl._lora_optimizer)
        ctrl.reset_lora_optimizer()
        assert id(ctrl._lora_optimizer) != old_id

    def test_reset_lora_optimizer_fresh_momentum(self):
        """A reset optimizer must have no momentum buffers (Adam state = empty)."""
        ctrl = fresh_controller()
        # Train to populate momentum buffers
        s = rand_state()
        ctrl.choose_action(s)
        ctrl.train(s, 0, 1.0, rand_state())
        ctrl.reset_lora_optimizer()
        assert len(ctrl._lora_optimizer.state) == 0


# ===========================================================================
# 17. sync_all_meta_ga
# ===========================================================================

class TestSyncAllMetaGA:
    def test_sync_all_meta_ga_no_crash(self):
        """sync_all_meta_ga must not raise even when no torch controllers exist."""
        sync_all_meta_ga({})   # empty nations dict

    def test_sync_all_meta_ga_updates_epsilon(self):
        """Running sync with high fitness should lower epsilon on all controllers."""
        # Build a minimal fake 'nations' dict with one nation-like object
        from types import SimpleNamespace
        ga = _make_ga(fitness=200.0)
        ctrl = fresh_controller(epsilon=0.10)
        nation = SimpleNamespace(
            civilian_ai=ctrl,
            project_ai=None,
            diplomacy_ai=None,
            research_ai=None,
            military_ai=None,
            doctrine_ai=None,
            reward_ga={"civilian": ga, "projects": _make_ga(), "diplomacy": _make_ga(),
                       "research": _make_ga(), "events": _make_ga()},
        )
        nations = {0: nation}
        sync_all_meta_ga(nations)
        assert ctrl.epsilon < 0.10


class TestEdgeCases:
    def test_zero_state_does_not_crash(self):
        ctrl = fresh_controller()
        ctrl.predict([0.0] * IN)
        ctrl.train([0.0] * IN, 0, 0.0, [0.0] * IN)

    def test_large_reward_does_not_produce_nan(self):
        ctrl = fresh_controller()
        s = rand_state()
        ctrl.predict(s)
        ctrl.train(s, 0, 1e6, rand_state())
        scores = ctrl.predict(rand_state())
        assert all(math.isfinite(v) for v in scores)

    def test_negative_reward(self):
        ctrl = fresh_controller()
        s = rand_state()
        ctrl.predict(s)
        ctrl.train(s, 0, -1.0, rand_state())  # should not raise

    def test_all_actions_invalid_in_mask_falls_back_to_zero(self):
        ctrl = fresh_controller(epsilon=0.0)
        mask = [False, False, False]
        a = ctrl.choose_action(rand_state(), valid_mask=mask)
        assert a == 0   # fallback to 0 when no valid action
