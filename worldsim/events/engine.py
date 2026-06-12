"""Event engine with contextual Q-learning and heuristic fallback.

Each :class:`EventDecisionEngine` instance carries an
:class:`~worldsim.events.qlearner.EventQLearner` that learns nation-context →
action mappings across the lifetime of a simulation run.  Q-tables are
persisted to YAML files at run end and reloaded the next time the engine
starts, enabling progressive improvement across runs.

Design notes
------------
* State is coarsely bucketed (economy × stability × at_war) to keep tables
  compact while still capturing the most important decision context.
* When no Q-data exists for an (event, state) pair the engine falls back to
  static heuristic scoring, so first-run behaviour is sensible.
* Memory is updated every event regardless of the ``collect`` flag so the
  learner improves even in summary-only mode.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

try:
    import yaml as _yaml          # noqa: F401 — used for Q-table I/O
except ImportError:               # pragma: no cover
    _yaml = None

from ..planets import PLANETS
from .generator import ProceduralEventGenerator
from .qlearner import EventQLearner
from .rust_bridge import RustEventBridge

if TYPE_CHECKING:  # pragma: no cover
    from ..nations import Nation


class EventDecisionEngine:
    """Event resolver backed by a contextual Q-learner with heuristic fallback."""

    def __init__(
        self,
        events: Dict[str, List[Dict]],
        *,
        qtable_path: str | None = None,
        use_rust: bool | None = None,
        **_: object,
    ) -> None:
        self.events = events
        self.proc = ProceduralEventGenerator()
        self.qtable_path = Path(qtable_path) if qtable_path else None
        self.q_learner  = EventQLearner()
        if self.qtable_path:
            self.load_qtables()
        # Optional rust_events crate backend.  Opt-in: it round-trips its own
        # YAML table through a subprocess per event, so ``None`` only enables
        # it when WORLDSIM_USE_RUST_EVENTS requests it explicitly.
        self.rust_bridge: RustEventBridge | None = None
        if use_rust is None:
            use_rust = os.getenv("WORLDSIM_USE_RUST_EVENTS", "0").lower() in {"1", "true", "yes"}
        if use_rust and self.qtable_path:
            self.rust_bridge = RustEventBridge.create(self.qtable_path)

    # ------------------------------------------------------------------
    # Q-table persistence
    # ------------------------------------------------------------------

    def _qtable_path_for(self, event_name: str) -> Optional[Path]:
        """Return the YAML path for *event_name*’s Q-table."""
        if not self.qtable_path:
            return None
        base = str(self.qtable_path)
        if base.lower().endswith(".yml"):
            base = base[:-4]
        safe = event_name.replace(" ", "_").replace("/", "-")
        return Path(f"{base}_{safe}.yml")

    def save_qtables(self) -> None:
        """Persist learned Q-table data to per-event YAML files.

        Each file stores the state-keyed action-value table for one event
        type.  Existing files are overwritten with the latest values so the
        next run loads the accumulated knowledge.
        """
        if not self.qtable_path or _yaml is None:
            return
        # Write tables for every tracked event (including procedural ones)
        all_events = set(self.events) | set(self.q_learner.tables)
        for name in all_events:
            path = self._qtable_path_for(name)
            if path is None:
                continue
            q_data = self.q_learner.tables.get(name, {})
            payload = {"event": name, "q_table": q_data}
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "w", encoding="utf8") as fh:
                    _yaml.safe_dump(payload, fh, default_flow_style=False)
            except OSError:
                pass

    def load_qtables(self) -> None:
        """Load Q-table data from any per-event YAML files found on disk."""
        if not self.qtable_path or _yaml is None:
            return
        base_path = Path(str(self.qtable_path))
        if base_path.suffix.lower() == ".yml":
            base_path = base_path.with_suffix("")
        search_dir = base_path.parent
        prefix     = base_path.name
        for path in search_dir.glob(f"{prefix}_*.yml"):
            try:
                with open(path, "r", encoding="utf8") as fh:
                    data = _yaml.safe_load(fh) or {}
            except OSError:
                continue
            event_name = data.get("event")
            q_table    = data.get("q_table")
            if not isinstance(event_name, str) or not isinstance(q_table, dict):
                continue
            self.q_learner.tables[event_name] = {
                key: [float(v) for v in vals]
                for key, vals in q_table.items()
                if isinstance(vals, list)
            }

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------
    def _state_snapshot(self, nation) -> List[float]:
        return [
            round(nation.economy, 3),
            round(nation.technology.overall, 3),
            round(nation.military, 3),
            round(nation.infrastructure, 3),
            round(nation.stability, 3),
        ]

    def _trigger_probability(self, nation) -> float:
        base = 0.1
        base += max(0.0, (50 - nation.stability) / 200)
        if nation.economy_linear < 150:
            base += 0.05
        if nation.infrastructure < 40:
            base += 0.05
        return max(0.0, min(base, 0.6))

    def _score_option(self, nation, option: Dict) -> float:
        effects = option.get("effects", {})
        weights = {
            "stability": 2.0,
            "economy": 1.5,
            "science": 1.2,
            "population": 1.0,
            "military": 1.0,
            "plague": -40.0,
            "radiation": -20.0,
        }
        score = 0.0
        for key, value in effects.items():
            weight = weights.get(key, 0.3)
            score += weight * value
        return score

    def _choose_option(self, nation, event: str, options: List[Dict]) -> int:
        """Pick the best option, consulting the Q-learner first.

        When the Q-learner has data for the current (event, state) pair it
        takes precedence (with epsilon-greedy exploration).  Otherwise the
        heuristic scorer is used as a sensible default.
        """
        if not options:
            return 0
        if self.rust_bridge is not None and self.rust_bridge.alive:
            rust_choice = self.rust_bridge.choose(
                event, len(options), self._state_snapshot(nation)
            )
            if rust_choice is not None:
                return rust_choice
        q_choice = self.q_learner.choose(nation, event, len(options))
        if q_choice is not None:
            return q_choice
        # Heuristic fallback
        scored = [(self._score_option(nation, opt), idx) for idx, opt in enumerate(options)]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return scored[0][1]

    def pick_event(self, nation, nations: Dict = None) -> Tuple[str, List[Dict]]:
        # Procedural first — contextually appropriate
        if random.random() < 0.8:
            return self.proc.generate(nation, nations or {})
    
        # Static events as flavor — but filter out punishing ones 
        # when nation is already struggling
        available = list(self.events.keys())
        if nation.stability < 40:
            # Don't pile on struggling nations with political scandals
            available = [e for e in available 
                        if e not in ("political scandal",)]
    
        name = random.choice(available)
        return name, self.events[name]

    # ------------------------------------------------------------------
    # Effect application
    # ------------------------------------------------------------------
    def apply_effects(
        self, nation, effects: Dict[str, float], event: str | None = None
    ) -> None:
        if event is not None and effects.get("stability", 0.0) < 0:
            count = nation.event_history.get(event, 0) + 1
            nation.event_history[event] = count
            if count >= 3:
                effects = effects.copy()
                effects["stability"] *= count - 1
        for key, val in effects.items():
            if key == "population" and nation.cities:
                change = val / len(nation.cities)
                for c in nation.cities:
                    c.population = int(max(0, c.population + change))
                nation.population += val
            elif key == "plague":
                planet = PLANETS.get(nation.planet)
                if planet:
                    planet.plague_level = max(0.0, planet.plague_level + val)
            elif key == "radiation":
                planet = PLANETS.get(nation.planet)
                if planet:
                    planet.radiation_level = max(0.0, planet.radiation_level + val)
            elif hasattr(nation, key):
                setattr(nation, key, getattr(nation, key) + val)
            elif hasattr(nation.technology, key):
                setattr(
                    nation.technology,
                    key,
                    getattr(nation.technology, key) + val,
                )

    def _resolve_choice(
        self,
        nation,
        event: str,
        options: List[Dict],
        idx: int,
        collect: bool,
    ):
        """Apply the selected option’s effects, train the Q-learner, and return."""
        if not options:
            return None
        idx = max(0, min(idx, len(options) - 1))
        selection = options[idx]
        choice  = selection.get("choice", "")
        effects = selection.get("effects", {})

        before = self._state_snapshot(nation)
        self.apply_effects(nation, effects, event)
        after  = self._state_snapshot(nation)

        # Compute reward regardless of collect — needed for Q-learning.
        if hasattr(nation, "compute_reward"):
            try:
                reward = float(nation.compute_reward("events", before, after))
            except Exception:
                reward = sum(a - b for a, b in zip(after, before))
        else:
            reward = sum(a - b for a, b in zip(after, before))

        # Always update the Q-learner so it learns from every event.
        self.q_learner.update(nation, event, idx, reward, len(options))
        # Mirror the transition into the rust_events table when the bridge is
        # active, so both backends learn from the same history.
        if self.rust_bridge is not None and self.rust_bridge.alive:
            self.rust_bridge.update(event, before, idx, reward, after)

        if collect:
            return {
                "event":       event,
                "options":     [o.get("choice", "") for o in options],
                "choice_index": idx,
                "choice":      choice,
                "effects":     effects,
                "state_before": before,
                "state_after":  after,
                "reward":      reward,
            }
        return f"{event}: {choice}"

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def run_event(self, nation, *, collect: bool = False):
        event, opts = self.pick_event(nation)
        idx = self._choose_option(nation, event, opts)
        return self._resolve_choice(nation, event, opts, idx, collect)

    def run_event_batch(
        self, nations: List["Nation"], *, collect: bool = False
    ) -> List[object]:
        return [self.run_event(n, collect=collect) for n in nations]

    def maybe_trigger_event_batch(
        self, nations: List["Nation"], *, collect: bool = False
    ) -> List[object | None]:
        results: List[object | None] = [None] * len(nations)
        for idx, nation in enumerate(nations):
            if random.random() < self._trigger_probability(nation):
                results[idx] = self.run_event(nation, collect=collect)
        return results

    def maybe_trigger_event(self, nation, *, collect: bool = False):
        if random.random() < self._trigger_probability(nation):
            return self.run_event(nation, collect=collect)
        return None

    def force_event(
        self,
        nation,
        event: str,
        choice_index: int = 0,
        *,
        collect: bool = False,
    ):
        opts = self.events.get(event)
        if not opts:
            return None
        return self._resolve_choice(nation, event, opts, choice_index, collect)
