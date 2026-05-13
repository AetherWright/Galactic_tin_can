"""Event engine with contextual Q-learning and heuristic fallback.

Each :class:`EventDecisionEngine` instance carries an :class:`EventQLearner`
that learns nation-context → action mappings across the lifetime of a
simulation run.  Q-tables are persisted to YAML files at run end and reloaded
the next time the engine starts, enabling progressive improvement across runs.

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
from typing import Dict, List, Optional, Tuple

try:
    import yaml as _yaml          # noqa: F401 — used for Q-table I/O
except ImportError:               # pragma: no cover
    _yaml = None

from .planets import PLANETS


# ---------------------------------------------------------------------------
# EventQLearner — contextual action-value table
# ---------------------------------------------------------------------------

class EventQLearner:
    """Contextual Q-table for event response learning.

    State context is bucketed into 18 bins to keep the tables small
    while still capturing economy, stability, and war status:

    * ``economy_level``  – 0 (< 200), 1 (< 600), 2 (≥ 600)
    * ``stability_level`` – 0 (< 40),  1 (< 70),  2 (≥ 70)
    * ``at_war``         – 0 / 1

    Each (event, state_key) → list of Q-values, one per choice.
    Updates use simple TD(0):

        Q[choice] += ALPHA * (reward - Q[choice])
    """

    ALPHA:   float = 0.25    # learning rate — fast enough for short runs
    EPSILON: float = 0.15    # exploration rate

    def __init__(self) -> None:
        # tables[event][state_key] = [q_val_per_choice]
        self.tables: Dict[str, Dict[str, List[float]]] = {}

    # ------------------------------------------------------------------
    # State bucketing
    # ------------------------------------------------------------------

    @staticmethod
    def _state_key(nation) -> str:
        econ = (0 if nation.economy_linear < 200
                else (1 if nation.economy_linear < 600 else 2))
        stab = (0 if nation.stability < 40
                else (1 if nation.stability < 70 else 2))
        war  = 1 if nation.at_war else 0
        return f"{econ},{stab},{war}"

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def choose(self, nation, event: str, n_choices: int) -> Optional[int]:
        """Return the best action index, or ``None`` if the table is empty.

        Falls back to the caller’s heuristic when no data exists for the
        current (event, state) pair.
        """
        if n_choices <= 0:
            return None
        key   = self._state_key(nation)
        q_map = self.tables.get(event)
        if not q_map:
            return None
        q_vals = q_map.get(key)
        if not q_vals:
            return None
        if random.random() < self.EPSILON:
            return random.randrange(n_choices)
        # Pad to n_choices in case the table was built with a different count
        q = (list(q_vals) + [0.0] * n_choices)[:n_choices]
        return q.index(max(q))

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def update(
        self,
        nation,
        event:    str,
        choice:   int,
        reward:   float,
        n_choices: int,
    ) -> None:
        """TD(0) update: Q[choice] += ALPHA * (reward - Q[choice])."""
        if n_choices <= 0:
            return
        key   = self._state_key(nation)
        table = self.tables.setdefault(event, {})
        q_vals = table.get(key)
        if q_vals is None:
            q_vals = [0.0] * n_choices
            table[key] = q_vals
        # Grow to accommodate choice index if needed
        while len(q_vals) < n_choices:
            q_vals.append(0.0)
        idx = max(0, min(choice, n_choices - 1))
        q_vals[idx] += self.ALPHA * (reward - q_vals[idx])

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Dict[str, List[float]]]:
        return {
            event: dict(entries)
            for event, entries in self.tables.items()
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "EventQLearner":
        obj = cls()
        for event, entries in data.items():
            if not isinstance(entries, dict):
                continue
            obj.tables[event] = {
                key: [float(v) for v in vals]
                for key, vals in entries.items()
                if isinstance(vals, list)
            }
        return obj

EVENTS = {
    "plague": [
        {
            "choice": "Quarantine affected cities",
            "effects": {"plague": -0.02, "stability": -5},
        },
        {"choice": "Ignore the plague", "effects": {"plague": 0.02}},
        {"choice": "Seek foreign aid", "effects": {"economy": 10, "plague": -0.01}},
    ],
    "gold rush": [
        {"choice": "Nationalize resources", "effects": {"economy": 20, "stability": 3}},
        {"choice": "Open to investors", "effects": {"economy": 30, "stability": -3}},
    ],
    "tech breakthrough": [
        {"choice": "Invest heavily in research", "effects": {"science": 5, "economy": -10}},
        {"choice": "Patent and sell", "effects": {"economy": 15, "stability": 2}},
    ],
    "political scandal": [
        {"choice": "Cover it up", "effects": {"stability": -10}},
        {"choice": "Public inquiry", "effects": {"stability": -5, "economy": -5}},
        {"choice": "Resign gracefully", "effects": {"stability": 3, "economy": -3}},
     ]
}

class ProceduralEventGenerator:
    """Create simple events based on a nation's state."""

    def generate(self, nation, nations: Dict) -> Tuple[str, List[Dict]]:
        """Generate contextually appropriate events based on actual nation state."""
        
        # Build situation profile
        at_war = len(nation.at_war) > 0
        neighbors_at_war = any(
            len(n.at_war) > 0 for nid, n in nations.items() 
            if nid in nation.relations
        )
        high_tech = nation.technology.science > 30
        struggling = nation.economy_linear < 100
        large = len(nation.cities) > 5
        unstable = nation.stability < 40
        prosperous = nation.economy_linear > 300 and nation.stability > 60
        has_nukes = "Nuclear Weapons" in nation.tech_tree.unlocked
        plague_nearby = any(
            PLANETS.get(c.planet) and PLANETS[c.planet].plague_level > 0.1
            for c in nation.cities
        )
        
        # Weighted event pool based on actual situation
        pool = []
   
        if at_war:
            pool += [
                ("supply shortage", [
                    {"choice": "Ration supplies", "effects": {"economy": -15, "stability": -3}},
                    {"choice": "Emergency production", "effects": {"economy": -25, "stability": 5}},
                ]),
                ("war hero emerges", [
                    {"choice": "Promote publicly", "effects": {"stability": 10, "military": 5}},
                    {"choice": "Keep quiet", "effects": {"military": 3}},
                ]),
                ("desertion crisis", [
                    {"choice": "Execute deserters", "effects": {"stability": -10, "military": 5}},
                    {"choice": "Offer amnesty", "effects": {"stability": 5, "military": -5}},
                ]),
            ] * 3  # weight war events heavily when at war
    
        if plague_nearby or any(
            getattr(PLANETS.get(c.planet), 'plague_level', 0) > 0 
            for c in nation.cities
        ):
            pool += [
                ("epidemic spreading", [
                    {"choice": "Strict quarantine", "effects": {"stability": -8, "economy": -10}},
                    {"choice": "Develop treatment", "effects": {"economy": -20, "science": 3}},
                    {"choice": "Ignore", "effects": {"plague": 0.05}},
                ]),
            ] * 2
    
        if high_tech and not at_war:
            pool += [
                ("research breakthrough", [
                    {"choice": "Publish findings", "effects": {"science": 8, "stability": 3}},
                    {"choice": "Classify research", "effects": {"science": 4, "military": 5}},
                    {"choice": "Sell to industry", "effects": {"economy": 20, "science": 2}},
                ]),
                ("automation displacement", [
                    {"choice": "Retrain workers", "effects": {"economy": -15, "stability": 5}},
                    {"choice": "Let market adjust", "effects": {"economy": 10, "stability": -8}},
                    {"choice": "Regulate automation", "effects": {"economy": -5, "stability": 3}},
                ]),
            ]
        
        if has_nukes:
            pool += [
                ("proliferation pressure", [
                    {"choice": "Sign treaty", "effects": {"stability": 5, "military": -3}},
                    {"choice": "Refuse", "effects": {"stability": -5, "military": 5}},
                    {"choice": "Share technology", "effects": {"economy": 20, "stability": -8}},
                ]),
            ]
    
        if large and not at_war:
            pool += [
                ("administrative reform", [
                    {"choice": "Centralize", "effects": {"stability": -5, "economy": 10}},
                    {"choice": "Decentralize", "effects": {"stability": 8, "economy": -5}},
                    {"choice": "Status quo", "effects": {}},
                ]),
                ("separatist movement", [
                    {"choice": "Negotiate autonomy", "effects": {"stability": -3, "economy": -5}},
                    {"choice": "Crack down", "effects": {"stability": -12, "military": 3}},
                    {"choice": "Grant independence", "effects": {"stability": 5, "economy": -15}},
                ]),
            ]
        
        if prosperous:
            pool += [
                ("golden age", [
                    {"choice": "Invest in culture", "effects": {"stability": 12, "economy": -5}},
                    {"choice": "Expand military", "effects": {"military": 10, "stability": 3}},
                    {"choice": "Fund exploration", "effects": {"economy": -10, "science": 5}},
                ]),
                ("trade boom", [
                    {"choice": "Open borders", "effects": {"economy": 25, "stability": 5}},
                    {"choice": "Protect industry", "effects": {"economy": 10, "stability": 3}},
                ]),
            ] * 2
    
        if unstable:
            pool += [
                ("reform movement", [
                    {"choice": "Embrace reform", "effects": {"stability": 10, "economy": -10}},
                    {"choice": "Suppress", "effects": {"stability": -10, "military": 3}},
                    {"choice": "Compromise", "effects": {"stability": 5, "economy": -5}},
                ]),
                ("coup attempt", [
                    {"choice": "Crush it", "effects": {"stability": -5, "military": 5}},
                    {"choice": "Negotiate", "effects": {"stability": 3, "economy": -5}},
                    {"choice": "Flee", "effects": {"stability": -20}},
                ]),
            ] * 2
        
        if struggling:
            pool += [
                ("debt crisis", [
                    {"choice": "Austerity", "effects": {"economy": 15, "stability": -10}},
                    {"choice": "Default", "effects": {"economy": -10, "stability": -5}},
                    {"choice": "Foreign aid", "effects": {"economy": 10, "stability": 3}},
                ]),
                ("food shortage", [
                    {"choice": "Import food", "effects": {"economy": -15, "stability": 3}},
                    {"choice": "Ration", "effects": {"economy": -5, "stability": -8}},
                    {"choice": "Ignore", "effects": {"population": -1000, "stability": -10}},
                ]),
            ] * 2
    
        if neighbors_at_war and not at_war:
            pool += [
                ("refugee crisis", [
                    {"choice": "Accept refugees", "effects": {"population": 5000, "stability": -5}},
                    {"choice": "Close borders", "effects": {"stability": -3, "economy": -5}},
                    {"choice": "Exploit situation", "effects": {"economy": 15, "stability": -8}},
                ]),
                ("war profiteering opportunity", [
                    {"choice": "Sell arms", "effects": {"economy": 25, "stability": -5}},
                    {"choice": "Stay neutral", "effects": {"stability": 3}},
                    {"choice": "Mediate", "effects": {"stability": 5, "economy": -5}},
                ]),
            ]
    
            # Always have some baseline events as fallback
            pool += [
                ("census results", [
                    {"choice": "Publish data", "effects": {"stability": 2}},
                    {"choice": "Classify data", "effects": {"stability": -1}},
            ]),
            ("infrastructure aging", [
                {"choice": "Emergency repairs", "effects": {"economy": -10, "stability": 3}},
                {"choice": "Defer maintenance", "effects": {"stability": -3}},
            ]),
            ]
    
        if not pool:
            # Absolute fallback
            return "civil unrest", [
                {"choice": "Promise reforms", "effects": {"stability": 3}},
                {"choice": "Crack down", "effects": {"stability": -5}},
            ]
    
        name, choices = random.choice(pool)
        return name, choices
class EventDecisionEngine:
    """Event resolver backed by a contextual Q-learner with heuristic fallback."""

    def __init__(
        self,
        events: Dict[str, List[Dict]],
        *,
        qtable_path: str | None = None,
        **_: object,
    ) -> None:
        self.events = events
        self.proc = ProceduralEventGenerator()
        self.qtable_path = Path(qtable_path) if qtable_path else None
        self.q_learner  = EventQLearner()
        if self.qtable_path:
            self.load_qtables()

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
