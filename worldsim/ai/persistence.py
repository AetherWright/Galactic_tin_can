"""Utilities for saving, merging, and seeding AI models across simulation runs.

Two main workflows are supported:

1. **Per-nation tables** (``save_gp_model``)
   Persists each nation’s individual RL controller to its own YAML file and
   stores the averaged GA weights as a JSON sidecar.  Useful for analysing a
   specific run.

2. **Cross-run seed** (``merge_and_save_models`` / ``load_model_seeds``)
   After a run, all nations’ networks are averaged into a single “seed” per
   AI role and written to a seed directory.  At the start of the *next* run,
   ``load_model_seeds`` injects those averaged weights into every new nation,
   giving them a head start that improves with each generation of play.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from itertools import zip_longest
from pathlib import Path
from typing import Dict, List, Optional

try:
    import yaml as _yaml
except ImportError:          # pragma: no cover — yaml is a project requirement
    _yaml = None

from ..nations import Nation
from .meta_ga import RewardGA


# ---------------------------------------------------------------------------
# Role → attribute-name mapping for NelderMeadPolicy-based controllers
# ---------------------------------------------------------------------------

_ROLE_TO_ATTR: Dict[str, str] = {
    "military":  "military_ai",
    "civilian":  "civilian_ai",
    "projects":  "project_ai",
    "diplomacy": "diplomacy_ai",
    "research":  "research_ai",
    "doctrine":  "doctrine_ai",
    "events":    "events_ai",
}


# ---------------------------------------------------------------------------
# GA helpers  (unchanged from original)
# ---------------------------------------------------------------------------

def _merge_weights(gas: list[RewardGA]) -> list[float]:
    """Return average weight vector from multiple :class:`RewardGA` objects."""
    if not gas:
        return []
    length = len(gas[0].weights)
    totals = [0.0] * length
    for ga in gas:
        for i, w in enumerate(ga.weights[:length]):
            totals[i] += w
    return [val / len(gas) for val in totals]


# ---------------------------------------------------------------------------
# Network averaging
# ---------------------------------------------------------------------------

def _average_network_dict(net_dicts: List[Dict]) -> Dict:
    """Elementwise-average a list of :meth:`LayeredNetwork.to_dict` payloads.

    All entries must share the same architecture (``layer_sizes``).  Forward
    and backward weights are averaged in addition to the main weight tensors.
    Masks are taken from the first entry unchanged (they are structural and
    identical across instances initialised with the same seed pattern).
    """
    n = len(net_dicts)
    if n == 0:
        raise ValueError("empty network list")
    template = net_dicts[0]
    result   = {k: v for k, v in template.items()}   # shallow copy metadata

    # --- forward weights [layer][row][col] ---
    result["weights"] = [
        [
            [
                sum(nd["weights"][li][ri][ci] for nd in net_dicts) / n
                for ci in range(len(row))
            ]
            for ri, row in enumerate(layer)
        ]
        for li, layer in enumerate(template["weights"])
    ]

    # --- biases [layer][neuron] ---
    result["biases"] = [
        [
            sum(nd["biases"][li][ni] for nd in net_dicts) / n
            for ni in range(len(layer))
        ]
        for li, layer in enumerate(template["biases"])
    ]

    # --- backward weights (optional) ---
    bw = template.get("backward_weights", [])
    if bw:
        result["backward_weights"] = [
            [
                [
                    sum(nd["backward_weights"][li][ri][ci] for nd in net_dicts) / n
                    for ci in range(len(row))
                ]
                for ri, row in enumerate(layer)
            ]
            for li, layer in enumerate(bw)
        ]
    return result


# ---------------------------------------------------------------------------
# NelderMeadPolicy merging
# ---------------------------------------------------------------------------

def _build_merged_payload(policies: list) -> Optional[Dict]:
    """Produce a merged YAML payload from a list of NelderMeadPolicy objects.

    Only policies sharing the same (n_inputs, n_actions, hidden_layers)
    architecture are merged.  Memories are interleaved from all compatible
    policies for maximum diversity.
    Returns ``None`` if the list is empty or no compatible policies exist.

    Only NelderMeadPolicy-style controllers expose the
    ``(n_inputs, n_actions, network, memory, scales)`` interface this merger
    needs.  Torch RNN controllers and the hierarchical ``CivilianController``
    are persisted separately (see the ``torch_bases/`` handling in
    :func:`merge_and_save_models`), so anything lacking that interface is
    skipped rather than crashing the merge.
    """
    policies = [
        p for p in policies
        if hasattr(p, "n_actions") and hasattr(p, "network")
        and hasattr(p, "memory") and hasattr(p, "scales")
    ]
    if not policies:
        return None

    def _arch(p):
        return (p.n_inputs, p.n_actions, tuple(p.hidden_layers))

    counts  = Counter(_arch(p) for p in policies)
    target  = counts.most_common(1)[0][0]
    compat  = [p for p in policies if _arch(p) == target]
    if not compat:
        return None

    n = len(compat)
    avg_net = _average_network_dict([p.network.to_dict() for p in compat])

    # Average scales (use the shortest list to stay safe)
    scale_len  = min(len(p.scales) for p in compat)
    avg_scales = [sum(p.scales[i] for p in compat) / n for i in range(scale_len)]

    # Interleave memories so diverse experiences are represented
    max_mem  = max(p.memory.maxlen or 32 for p in compat)
    combined: List[Dict] = []
    for entries in zip_longest(*[list(p.memory) for p in compat]):
        for entry in entries:
            if entry is not None:
                s, t = entry
                combined.append({"state": list(s), "target": list(t)})
        if len(combined) >= max_mem:
            break
    random.shuffle(combined)
    combined = combined[:max_mem]

    return {
        "type":         "merged_seed",
        "n_inputs":     target[0],
        "n_actions":    target[1],
        "hidden_layers": list(target[2]),
        "epsilon":      sum(p.epsilon      for p in compat) / n,
        "gamma":        sum(p.gamma        for p in compat) / n,
        "hebbian_lr":   sum(p.hebbian_lr   for p in compat) / n,
        "hebbian_clip": sum(p.hebbian_clip for p in compat) / n,
        "scales":       avg_scales,
        "network":      avg_net,
        "memory_size":  max_mem,
        "memory":       combined,
    }


def _civilian_subgroups(nations: Dict[int, Nation]) -> Dict[str, list]:
    """Group civilian sub-models by their position in the hierarchy.

    Returns a mapping ``name -> [policy, ...]`` where *name* is ``"overseer"``
    or a department slug, gathering the matching sub-model from every nation
    whose ``civilian_ai`` is a :class:`CivilianController`.  Used so the seed
    merger can average each sub-model across nations independently.  Non-torch
    (NelderMead) and torch sub-models are returned alike; the actual merge in
    :func:`_build_merged_payload` keeps only the NelderMead-shaped ones (torch
    controllers are seeded via the shared ``torch_bases`` checkpoints instead).
    """
    from ..nations.civilian import CivilianController

    groups: Dict[str, list] = {}
    for n in nations.values():
        ctrl = getattr(n, "civilian_ai", None)
        if not isinstance(ctrl, CivilianController):
            continue
        groups.setdefault("overseer", []).append(ctrl.overseer)
        for slug, model in ctrl.dept_models.items():
            groups.setdefault(slug, []).append(model)
    return groups


def _load_civilian_seeds(nations: Dict[int, Nation], seed_dir: Path) -> bool:
    """Inject merged civilian sub-model seeds back into each nation.

    Mirrors :func:`_civilian_subgroups`: looks for ``seed_civilian_{name}.yml``
    files and applies each to the corresponding overseer / department model.
    Returns ``True`` if at least one seed file was applied.
    """
    from ..nations.civilian import CivilianController

    loaded = False
    cache: Dict[str, Optional[Dict]] = {}

    def _payload_for(name: str) -> Optional[Dict]:
        if name not in cache:
            path = seed_dir / f"seed_civilian_{name}.yml"
            if not path.exists():
                cache[name] = None
            else:
                try:
                    with open(path, encoding="utf8") as fh:
                        cache[name] = _yaml.safe_load(fh) or {}
                except OSError:
                    cache[name] = None
        return cache[name]

    for nation in nations.values():
        ctrl = getattr(nation, "civilian_ai", None)
        if not isinstance(ctrl, CivilianController):
            continue
        for name, model in (("overseer", ctrl.overseer), *ctrl.dept_models.items()):
            payload = _payload_for(name)
            if not payload:
                continue
            try:
                _apply_seed_payload(model, payload)
                loaded = True
            except Exception:    # architecture mismatch / torch sub-model
                pass
    return loaded


def _apply_seed_payload(policy, payload: Dict) -> None:
    """Load averaged seed weights into an existing NelderMeadPolicy.

    Skips silently when architectures don’t match.  Does **not** touch
    ``policy.table_path`` so the nation’s own per-run saves remain intact.
    """
    from .networks import LayeredNetwork

    if (payload.get("n_inputs")  != policy.n_inputs or
            payload.get("n_actions") != policy.n_actions):
        return

    network_data = payload.get("network")
    if isinstance(network_data, dict):
        try:
            policy.network = LayeredNetwork.from_dict(network_data)
        except Exception:
            return

    scales = payload.get("scales")
    if isinstance(scales, list) and scales:
        expected = policy._expected_scale_length()
        parsed   = [float(v) for v in scales]
        if len(parsed) < expected:
            parsed.extend([1.0] * (expected - len(parsed)))
        policy.scales = parsed[:expected]

    for attr in ("epsilon", "gamma", "hebbian_lr", "hebbian_clip"):
        val = payload.get(attr)
        if isinstance(val, (int, float)):
            setattr(policy, attr, float(val))

    # Seed the replay buffer with merged experiences
    policy.memory.clear()
    for entry in payload.get("memory", []):
        if not isinstance(entry, dict):
            continue
        s = entry.get("state")
        t = entry.get("target")
        if isinstance(s, list) and isinstance(t, list):
            policy.memory.append((s, t))


# ---------------------------------------------------------------------------
# Fleet controller merging  (LayeredNetworkFleetController)
# ---------------------------------------------------------------------------

def _build_merged_fleet_payload(controllers: list) -> Optional[Dict]:
    """Average fleet controller networks and scales."""
    if not controllers:
        return None
    def _arch(c):
        return (c.n_inputs, c.n_outputs)
    counts = Counter(_arch(c) for c in controllers)
    target = counts.most_common(1)[0][0]
    compat = [c for c in controllers if _arch(c) == target]
    if not compat:
        return None
    n   = len(compat)
    avg_net = _average_network_dict([c.network.to_dict() for c in compat])
    scale_len  = min(len(c.scales) for c in compat)
    avg_scales = [sum(c.scales[i] for c in compat) / n for i in range(scale_len)]
    return {
        "type":     "fleet_seed",
        "n_inputs": target[0],
        "n_outputs": target[1],
        "network": avg_net,
        "scales":  avg_scales,
    }


def _apply_fleet_seed(controller, payload: Dict) -> None:
    """Apply averaged fleet seed to an existing LayeredNetworkFleetController."""
    from .networks import LayeredNetwork
    if (payload.get("n_inputs")  != controller.n_inputs or
            payload.get("n_outputs") != controller.n_outputs):
        return
    net = payload.get("network")
    if isinstance(net, dict):
        try:
            controller.network = LayeredNetwork.from_dict(net)
        except Exception:
            return
    scales = payload.get("scales")
    if isinstance(scales, list) and scales:
        controller.scales = [float(v) for v in scales][:len(controller.scales)]


# ---------------------------------------------------------------------------
# GA seeding helpers
# ---------------------------------------------------------------------------

def _save_ga_seed(nations: Dict[int, Nation], path: Path) -> None:
    """Save merged leader/culture GA weights to JSON.

    (Directorate reward weighting is no longer GA-evolved — see
    ``Nation.compute_reward`` — so only the unrelated ``LeaderModel`` culture
    GA is seeded here now.)
    """
    info: Dict = {"leader": _merge_weights([n.leader_model.ga for n in nations.values()])}
    with open(path, "w", encoding="utf8") as fh:
        json.dump(info, fh)


def _apply_ga_seed(nations: Dict[int, Nation], path: Path) -> None:
    """Inject merged leader/culture GA weights into all nations."""
    if not path.exists():
        return
    with open(path, encoding="utf8") as fh:
        data = json.load(fh)
    leader_w = data.get("leader", [])
    for nation in nations.values():
        if leader_w and hasattr(nation, "leader_model"):
            n_w = nation.leader_model.ga.length
            lw  = (list(leader_w) + [1.0] * n_w)[:n_w]
            for genome in nation.leader_model.ga.population:
                genome.weights = list(lw)


# ---------------------------------------------------------------------------
# Public high-level API
# ---------------------------------------------------------------------------

def merge_and_save_models(
    nations:  Dict[int, Nation],
    seed_dir: str | Path,
) -> None:
    """Merge all nations’ AI models and write seed files for the next run.

    Files written to *seed_dir*:

    ==================  =====================================================
    ``seed_{role}.yml`` Averaged NelderMeadPolicy per role (military,
                        civilian, projects, diplomacy, research, doctrine)
    ``seed_fleet.yml``  Averaged fleet controller across all active fleets
    ``seed_ga.json``    Averaged RewardGA / LeaderModel weights
    ==================  =====================================================

    Each subsequent call overwrites the previous seed, so the directory
    always contains the freshest knowledge from the latest run.
    """
    if _yaml is None:
        return     # pragma: no cover
    seed_dir = Path(seed_dir)
    seed_dir.mkdir(parents=True, exist_ok=True)

    # --- per-role NelderMeadPolicy seeds ---
    for role, attr in _ROLE_TO_ATTR.items():
        # The civilian AI is a hierarchical CivilianController, not a flat
        # policy — merge its overseer + per-department sub-models separately.
        if role == "civilian":
            for name, policies in _civilian_subgroups(nations).items():
                payload = _build_merged_payload(policies)
                if payload is None:
                    continue
                with open(seed_dir / f"seed_civilian_{name}.yml", "w",
                          encoding="utf8") as fh:
                    _yaml.safe_dump(payload, fh, default_flow_style=False)
            continue
        policies = [
            getattr(n, attr)
            for n in nations.values()
            if getattr(n, attr, None) is not None
        ]
        payload = _build_merged_payload(policies)
        if payload is None:
            continue
        with open(seed_dir / f"seed_{role}.yml", "w", encoding="utf8") as fh:
            _yaml.safe_dump(payload, fh, default_flow_style=False)

    # --- fleet controller seed ---
    fleet_ctls = [
        f.controller
        for n in nations.values()
        for f in n.fleets
        if f.controller is not None and hasattr(f.controller, "network")
    ]
    fleet_payload = _build_merged_fleet_payload(fleet_ctls)
    if fleet_payload is not None:
        with open(seed_dir / "seed_fleet.yml", "w", encoding="utf8") as fh:
            _yaml.safe_dump(fleet_payload, fh, default_flow_style=False)

    # --- torch RNN base models (shared backbone per role × dims) ---
    # The PyTorch controllers don't fit the NelderMead seed format; persist
    # their shared base models as a directory of .pt checkpoints instead.
    try:
        from .rnn import rnn_available, save_all_base_models
        if rnn_available():
            save_all_base_models(seed_dir / "torch_bases")
    except Exception:
        pass

    # --- GA seed ---
    _save_ga_seed(nations, seed_dir / "seed_ga.json")


def load_model_seeds(
    nations:  Dict[int, Nation],
    seed_dir: str | Path,
) -> bool:
    """Inject merged seed weights into all nations’ AI controllers.

    Call this *after* :func:`~worldsim.galaxy.genesis.init_world` and
    *before* the first simulation step.  Nations will still diverge during
    the run (epsilon-greedy + different histories) but they start from the
    collective wisdom of the previous generation.

    Returns ``True`` if at least one seed file was found and applied.
    """
    if _yaml is None:
        return False      # pragma: no cover
    seed_dir = Path(seed_dir)
    if not seed_dir.exists():
        return False

    loaded = False

    # --- NelderMeadPolicy roles ---
    for role, attr in _ROLE_TO_ATTR.items():
        if role == "civilian":
            if _load_civilian_seeds(nations, seed_dir):
                loaded = True
            continue
        path = seed_dir / f"seed_{role}.yml"
        if not path.exists():
            continue
        try:
            with open(path, encoding="utf8") as fh:
                payload = _yaml.safe_load(fh) or {}
        except OSError:
            continue
        for nation in nations.values():
            policy = getattr(nation, attr, None)
            if policy is None:
                continue
            try:
                _apply_seed_payload(policy, payload)
                loaded = True
            except Exception:    # architecture mismatch or corrupt file
                pass

    # --- fleet controllers ---
    fleet_path = seed_dir / "seed_fleet.yml"
    if fleet_path.exists():
        try:
            with open(fleet_path, encoding="utf8") as fh:
                fleet_payload = _yaml.safe_load(fh) or {}
            for nation in nations.values():
                for fleet in nation.fleets:
                    ctrl = fleet.controller
                    if ctrl and hasattr(ctrl, "network"):
                        try:
                            _apply_fleet_seed(ctrl, fleet_payload)
                            loaded = True
                        except Exception:
                            pass
        except OSError:
            pass

    # --- torch RNN base models ---
    try:
        from .rnn import rnn_available, load_all_base_models
        bases_dir = seed_dir / "torch_bases"
        if rnn_available() and bases_dir.exists():
            load_all_base_models(bases_dir)
            loaded = True
    except Exception:
        pass

    # --- GA seed ---
    ga_path = seed_dir / "seed_ga.json"
    try:
        _apply_ga_seed(nations, ga_path)
        if ga_path.exists():
            loaded = True
    except Exception:
        pass

    return loaded


# ---------------------------------------------------------------------------
# Legacy save_gp_model  (kept for backwards compatibility)
# ---------------------------------------------------------------------------

def _collect_ai_controllers(nation: Nation) -> Dict[str, object]:
    """Return all reinforcement controllers attached to *nation*."""
    controllers = {
        "military": nation.military_ai,
        "civilian": nation.civilian_ai,
        "projects": nation.project_ai,
        "diplomacy": nation.diplomacy_ai,
        "research":  nation.research_ai,
        "events":    nation.events_ai,
    }
    return {name: ctrl for name, ctrl in controllers.items() if ctrl is not None}


def _save_ai_tables(nations: Dict[int, Nation], directory: Path) -> Dict[str, Dict[str, str]]:
    """Persist AI controller tables for *nations* into *directory*.

    Returns a manifest mapping nation names to controller filenames.
    """
    manifest: Dict[str, Dict[str, str]] = {}
    directory.mkdir(parents=True, exist_ok=True)
    for nation in nations.values():
        controllers = _collect_ai_controllers(nation)
        if not controllers:
            continue
        nation.ai_table_dir = directory
        entry: Dict[str, str] = {}
        for role, controller in controllers.items():
            table_path = nation._ai_table_path(role)
            if table_path is None:
                continue
            controller.save_table(table_path)
            entry[role] = table_path.name
        if entry:
            manifest[nation.name] = entry
    return manifest


def save_gp_model(
    nations: Dict[int, Nation],
    path: str | Path,
    *,
    ai_dir: str | Path | None = None,
) -> None:
    """Merge GA weights from all nations and persist them as JSON.

    Parameters
    ----------
    nations:
        Mapping of nation ids to :class:`~worldsim.nations.nation.Nation` objects.
    path:
        Output file for aggregated genetic policy weights.
    ai_dir:
        Optional directory for persisting nation-specific reinforcement learners.
        When omitted a folder named ``<path stem>_ai`` is created alongside
        ``path``.
    """
    info: Dict[str, object] = {
        "leader": _merge_weights([n.leader_model.ga for n in nations.values()])
    }

    path = Path(path)
    ai_directory = Path(ai_dir).resolve() if ai_dir else path.parent / f"{path.stem}_ai"
    manifest = _save_ai_tables(nations, ai_directory)
    if manifest:
        info["ai_tables"] = {
            "directory": str(ai_directory),
            "files": manifest,
        }

    with open(path, "w", encoding="utf8") as fh:
        json.dump(info, fh)
