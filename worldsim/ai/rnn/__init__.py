"""PyTorch RNN AI backend for WorldSim nation controllers.

Architecture
------------
One GRU-based *base model* is shared across all nations for each AI *role*
(war, civilian, project, diplomacy, research, doctrine, fleet).  Each nation
additionally owns a lightweight *LoRA adapter* per role that specialises the
shared backbone without touching its weights.

Rolling input buffer
    Every controller keeps a ``deque`` of the last :data:`BUFFER_SIZE`
    (= 5) input vectors.  At each forward pass the full buffer is fed to the
    GRU as a sequence, giving the model temporal context across simulation
    steps.

LoRA (Low-Rank Adaptation)
    The adapter modifies the input-projection and output-projection layers of
    the base GRU:

        eff_W_in  = W_in  + (B_in  @ A_in)  * (alpha / r)
        eff_W_out = W_out + (B_out @ A_out) * (alpha / r)

    ``A`` matrices are kaiming-uniform-initialised; ``B`` matrices start at
    zero, so the adapter has zero effect at initialisation and only diverges
    from the base model as training proceeds.

Training
    *LoRA*: Updated on every :meth:`RNNController.train` call using TD(0) and
    a per-nation Adam optimiser.  The backward pass is serialised through a
    per-role lock so concurrent nation threads do not corrupt shared base
    model parameter gradients.

    *Base model*: Each train call stores the experience in a replay buffer
    owned by the :class:`RoleBaseModel`.  Call
    :func:`step_all_base_models` from the main (non-threaded) simulation
    loop once per century to run gradient updates over the accumulated batch.

Public API (matches the NelderMeadPolicy surface)
    ``predict(state) -> List[float]``
    ``predict_prob(state) -> List[float]``
    ``choose_action(state, valid_mask=None) -> int``
    ``train(state, action, reward, next_state) -> None``
    ``save_table(path) / load_table(path)``

The package imports cleanly when PyTorch is missing: every public name is
then bound to ``None`` (classes) or a no-op (functions) and
:func:`rnn_available` returns ``False``.
"""
from __future__ import annotations

_TORCH_AVAILABLE: bool = False
try:
    import torch  # noqa: F401
    _TORCH_AVAILABLE = True
except ImportError:
    pass

# Number of past input states kept in the rolling buffer (also valid without torch).
BUFFER_SIZE: int = 5

# Doctrine and fleet-state labels (mirror military.doctrine / military.command;
# duplicated here to avoid a circular import).  Valid without torch.
_DOCTRINE_LIST = [
    "offensive", "defensive", "economic", "strategic_reserve", "total_war"
]
_FLEET_STATE_LIST = [
    "patrol", "assault", "defend", "escort", "retreat", "reposition", "colonize"
]

if _TORCH_AVAILABLE:
    from .base import (
        BUFFER_SIZE,
        RoleBaseModel,
        _BaseGRUModel,
        _DOCTRINE_LIST,
        _FLEET_STATE_LIST,
        _REGISTRY,
        _REGISTRY_LOCK,
        _ROLE_GA_KEY,
        get_role_model,
        load_all_base_models,
        save_all_base_models,
        step_all_base_models,
        sync_all_meta_ga,
    )
    from .controller import RNNController
    from .roles import (
        TorchCivilianOverseer,
        TorchDepartmentAI,
        TorchDiplomacyAI,
        TorchDoctrineAI,
        TorchDomesticPolicyAI,
        TorchFleetController,
        TorchProjectAI,
        TorchResearchAI,
        TorchWarAI,
    )
else:
    # -----------------------------------------------------------------------
    # Stubs — imported safely when torch is absent
    # -----------------------------------------------------------------------
    RNNController          = None   # type: ignore[assignment,misc]
    RoleBaseModel          = None   # type: ignore[assignment,misc]
    TorchWarAI             = None   # type: ignore[assignment,misc]
    TorchDomesticPolicyAI  = None   # type: ignore[assignment,misc]
    TorchCivilianOverseer  = None   # type: ignore[assignment,misc]
    TorchDepartmentAI      = None   # type: ignore[assignment,misc]
    TorchProjectAI         = None   # type: ignore[assignment,misc]
    TorchDiplomacyAI       = None   # type: ignore[assignment,misc]
    TorchResearchAI        = None   # type: ignore[assignment,misc]
    TorchDoctrineAI        = None   # type: ignore[assignment,misc]
    TorchFleetController   = None   # type: ignore[assignment,misc]

    def get_role_model(*_a: object, **_kw: object) -> None:   # type: ignore[return]
        return None

    def step_all_base_models(*_a: object, **_kw: object) -> None:
        pass

    def sync_all_meta_ga(*_a: object, **_kw: object) -> None:
        pass

    def save_all_base_models(*_a: object, **_kw: object) -> None:
        pass

    def load_all_base_models(*_a: object, **_kw: object) -> None:
        pass


def rnn_available() -> bool:
    """Return ``True`` if PyTorch is installed and the RNN backend is ready."""
    return _TORCH_AVAILABLE
