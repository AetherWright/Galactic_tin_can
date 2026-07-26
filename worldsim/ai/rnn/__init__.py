"""PyTorch RNN AI backend for WorldSim nation controllers.

Architecture
------------
**One shared GRU trunk** serves every AI role (war, civilian, project,
diplomacy, research, doctrine, fleet, events) — see :mod:`.unified` for the
full design. Every role's forward pass is built from two pieces:

    combined input = core(16, shared) ++ extra(role-specific, unchanged)

``core`` is :meth:`~worldsim.nations.nation.Nation.unified_core_state` — the
same economy/stability/military/tech/infrastructure/population/star/fleet
fields as ``reward_snapshot()``, NN-scaled, plus the nation's 8
:class:`~worldsim.society.culture.Culture` traits, so every role's decision
can condition on the nation's culture. ``extra`` is each role's own,
already-tuned feature vector, completely unchanged from before unification.

Each nation owns one :class:`~worldsim.ai.rnn.unified.UnifiedLoRABank`: a
*core* LoRA adapter (shared across every role for that nation — this is
where a nation's culture-driven personality lives) plus one small *head*
LoRA per role it uses (specialises that role's specific action preferences).
Neither the shared trunk's per-role ``extra_proj``/``role_embedding``/
``heads`` nor the GRU itself are LoRA'd — those are nation-invariant and
trained from a pooled replay buffer (:func:`~worldsim.ai.rnn.unified.step_trunk`,
called once per century from the main simulation loop).

Rolling input buffer
    Every role view keeps a ``deque`` of the last :data:`BUFFER_SIZE`
    (= 5) combined (core++extra) vectors, fed to the GRU as a sequence.

Training
    Double DQN (scalar heads) or C51 categorical cross-entropy (the
    ``events`` head) against a frozen *target* copy of the nation's LoRA,
    hard-updated every ``TARGET_UPDATE_FREQ`` steps — same as phase 1, just
    operating on the narrower LoRA surface described above.

Public API (unchanged from phase 1)
    ``predict(state) -> List[float]``
    ``predict_prob(state) -> List[float]``
    ``choose_action(state, valid_mask=None) -> int``
    ``train(state, action, reward, next_state) -> None``
    ``save_table(path) / load_table(path)``

The package imports cleanly when PyTorch is missing: every public name is
then bound to ``None`` (classes) or a no-op (functions) and
:func:`rnn_available` returns ``False`` — every role wrapper's caller falls
back to its legacy (non-torch) implementation, same convention as always.

History
-------
Phase 1 put every role on its own GRU trunk (one ``RoleBaseModel`` per role ×
input-dim × output-dim) plus a categorical events agent, and unified the
*reward function* (``Nation.compute_reward``/``reward_snapshot``) so every
role shares one designed reward instead of a GA-evolved one per directorate.

Phase 2 (this module) is the collapse those per-role trunks were designed to
enable: one shared trunk, culture folded into the input every role sees, and
the physics/engineering/biology research subsystems — previously still on
their original legacy (non-torch) ``ResearchAI`` — brought onto the same
unified trunk, retiring the dead, never-actually-called ``nation.research_ai``
in the process.
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
    from .unified import (
        BUFFER_SIZE,
        CORE_DIM,
        HIDDEN_DIM,
        N_ATOMS,
        UnifiedLoRABank,
        UnifiedRoleView,
        UnifiedTrunkModel,
        get_trunk,
        load_trunk,
        save_trunk,
        step_trunk,
    )
    from .roles import (
        TorchCivilianOverseer,
        TorchDepartmentAI,
        TorchDiplomacyAI,
        TorchDoctrineAI,
        TorchDomesticPolicyAI,
        TorchEventAI,
        TorchFleetController,
        TorchProjectAI,
        TorchResearchSubsystemAI,
        TorchWarAI,
    )
else:
    # -----------------------------------------------------------------------
    # Stubs — imported safely when torch is absent
    # -----------------------------------------------------------------------
    UnifiedRoleView        = None   # type: ignore[assignment,misc]
    UnifiedTrunkModel      = None   # type: ignore[assignment,misc]
    UnifiedLoRABank        = None   # type: ignore[assignment,misc]
    TorchWarAI             = None   # type: ignore[assignment,misc]
    TorchDomesticPolicyAI  = None   # type: ignore[assignment,misc]
    TorchCivilianOverseer  = None   # type: ignore[assignment,misc]
    TorchDepartmentAI      = None   # type: ignore[assignment,misc]
    TorchProjectAI         = None   # type: ignore[assignment,misc]
    TorchDiplomacyAI       = None   # type: ignore[assignment,misc]
    TorchResearchSubsystemAI = None   # type: ignore[assignment,misc]
    TorchDoctrineAI        = None   # type: ignore[assignment,misc]
    TorchFleetController   = None   # type: ignore[assignment,misc]
    TorchEventAI           = None   # type: ignore[assignment,misc]

    def get_trunk(*_a: object, **_kw: object) -> None:   # type: ignore[return]
        return None

    def step_trunk(*_a: object, **_kw: object) -> None:
        pass

    def save_trunk(*_a: object, **_kw: object) -> None:
        pass

    def load_trunk(*_a: object, **_kw: object) -> None:
        pass


def rnn_available() -> bool:
    """Return ``True`` if PyTorch is installed and the RNN backend is ready."""
    return _TORCH_AVAILABLE
