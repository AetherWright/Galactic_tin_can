"""The main century loop: ``run_simulation`` and ``SimulationLoop``."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

from ..ai.representations import build_alliance_matrix
from ..ai.rnn import step_trunk as _rnn_step_base_models
from ..core import flags, time_limit, vprint
from ..core.parallel import pooled_map, shutdown_pool
from ..diplomacy import (
    accumulate_war_exhaustion,
    check_war_victory,
    enforce_victory,
    update_war_score,
)
from ..events import EVENTS, EventDecisionEngine
from ..galaxy.genesis import add_star_systems
from ..military.combat import wage_war
from ..nations import Nation
from .blocs import update_border_pressures
from .filters import (
    _apply_great_filter,
    _cull_zombie_nations,
    _maybe_apply_soft_great_filter,
)
from .politics import _handle_internal_conflicts
from .reporting import _print_century_summary
from .territory import process_planets, update_star_ownership


def run_simulation(
    nations: Dict[int, Nation],
    turns: int | list[int] | None = 50,
    *,
    max_seconds: float | None = None,
    per_turn_timeout: bool = False,
    use_neat: bool | None = None,
    log_path: str | None = str(Path.cwd() / "sim_log.jsonl"),
    use_rust_events: bool | None = None,
    qtable_path: str = str(Path.cwd() / "events_q_table.yml"),
    human_events: list[dict] | None = None,
    control_path: str | None = None,
    console: bool = False,
    watch_nation: str | None = None,
) -> None:
    """Run the main simulation loop by century.

    Each century contains warfare subturns and five economic update phases.
    ``turns`` may be ``None`` to run indefinitely until interrupted.

    max_seconds:
        Optional wall-clock limit. When ``per_turn_timeout`` is ``False`` this
        applies to the overall run. With the flag enabled the limit resets for
        each turn so a single slow century cannot stall the entire simulation.
    per_turn_timeout:
        Apply ``max_seconds`` to each turn instead of the whole run.
    watch_nation:
        Name of a nation to show detailed output for.  All other nations appear
        in compact one-line summaries.  ``None`` shows compact rows for all.
    """

    # Propagate watch to model-level wprint helper
    flags.WATCHED_NATION = watch_nation

    def _run_war() -> None:
        matrix, _, index_map = build_alliance_matrix(nations)

        # Build set of active war pairs to avoid processing both directions
        processed: Set[Tuple[int, int]] = set()

        for n in nations.values():
            for enemy_id in list(n.at_war):
                pair = (min(n.id, enemy_id), max(n.id, enemy_id))
                if pair in processed:
                    continue
                processed.add(pair)

                enemy = nations.get(enemy_id)
                if not enemy:
                    continue

                # Fast resolve overwhelming superiority
                power_ratio = max(n.military, enemy.military) / max(1.0, min(n.military, enemy.military))
                if power_ratio > 10.0:
                    # Skip expensive calculations, just accumulate score
                    winner = n if n.military > enemy.military else enemy
                    loser = enemy if winner is n else n
                    update_war_score(winner, loser, kills_on_enemy=100, kills_on_self=0)
                    accumulate_war_exhaustion(loser, winner.id, 100)
                    if check_war_victory(winner, loser):
                        enforce_victory(winner, loser, nations)
                    continue

                row = matrix[index_map[n.id]] if index_map.get(n.id) is not None else None
                wage_war(n, nations, alliance_row=row)

    engine = EventDecisionEngine(
        EVENTS,
        use_neat=use_neat,
        use_rust=use_rust_events,
        qtable_path=qtable_path,
    )
    turns_ref: list[int | None]
    if isinstance(turns, list):
        turns_ref = turns
    else:
        turns_ref = [turns]
    century = 0
    if human_events is None:
        human_events = []
    recent_logs: List[tuple[str, list]] = []
    if log_path is not None:
        open(log_path, "w").close()
    console_ui = None
    if console:
        from ..interface.console import ConsoleUI
        console_ui = ConsoleUI([n.name for n in nations.values()])

    def _one_century() -> None:
        nonlocal century
        century += 1
        if century % 20 == 0:
            _apply_great_filter(nations)
        if control_path and Path(control_path).exists():
            import json

            with open(control_path) as fh:
                commands = json.load(fh)
            if isinstance(commands, dict):
                if "add_turns" in commands and turns_ref[0] is not None:
                    turns_ref[0] += int(commands["add_turns"])
                if "add_stars" in commands:
                    add_star_systems(int(commands["add_stars"]))
                if "events" in commands and isinstance(commands["events"], list):
                    human_events.extend(commands["events"])
            os.remove(control_path)
        _cull_zombie_nations(nations)
        for n in nations.values():
            n.step_meta(century)

        # Accumulate events across all 5 fifths; print summary once at the end.
        century_events: list = []

        def _process_one(n):
            n.process_turn(nations)
            _maybe_apply_soft_great_filter(n)

        for fifth in range(1, 6):
            _run_war()
            fifth_events: list = []

            nation_list = list(nations.values())
            pooled_map(_process_one, nation_list, mode="thread")
            descs = engine.maybe_trigger_event_batch(
                nation_list, collect=log_path is not None
            )

            for ev in list(human_events):
                try:
                    n = next(n for n in nation_list if n.name == ev.get("nation"))
                except StopIteration:
                    continue
                res = engine.force_event(
                    n,
                    ev.get("event", "plague"),
                    int(ev.get("choice", 0)),
                    collect=log_path is not None,
                )
                human_events.remove(ev)
                if res is not None:
                    if log_path is not None:
                        assert isinstance(res, dict)
                        fifth_events.append({"nation": n.name, **res, "manual": True})
                        if console_ui:
                            console_ui.append_nation_log(
                                n.name, f"{res['event']}: {res['choice']}"
                            )

            pooled_map(lambda n: n.finalize_turn(), nation_list, mode="thread")
            for nid, desc in zip([n.id for n in nation_list], descs):
                if desc is None:
                    continue
                name = nations[nid].name
                if log_path is not None:
                    assert isinstance(desc, dict)
                    fifth_events.append({"nation": name, **desc})
                else:
                    fifth_events.append({"nation": name, "event": str(desc)})

            _handle_internal_conflicts(nations, century)
            process_planets()
            update_star_ownership(nations)
            update_border_pressures(nations)

            century_events.extend(fifth_events)

            if log_path is not None:
                import json

                label = f"Century {century} F{fifth}"
                recent_logs.append((label, fifth_events))
                with open(log_path, "a") as fh:
                    json.dump({"century": century, "fifth": fifth, "events": fifth_events}, fh)
                    fh.write("\n")
                if len(recent_logs) > 20:
                    recent_logs.pop(0)

        # One compact summary per century (after fifth 5 completes)
        _print_century_summary(
            nations, century, 5,
            watch=watch_nation,
            events=century_events,
            console_ui=console_ui,
        )
        # Update shared RNN base models from accumulated replay buffers
        # (called from the main thread, not from threaded nation turns).
        _rnn_step_base_models()

    start_time = time.monotonic()
    try:
        if per_turn_timeout:
            while turns_ref[0] is None or century < turns_ref[0]:
                turn_start = time.monotonic()
                try:
                    with time_limit(max_seconds):
                        _one_century()
                except TimeoutError:
                    vprint("Turn runtime exceeded; stopping simulation.")
                    break
                if max_seconds is not None and time.monotonic() - turn_start >= max_seconds:
                    vprint("Turn runtime exceeded; stopping simulation.")
                    break
        else:
            with time_limit(max_seconds):
                while turns_ref[0] is None or century < turns_ref[0]:
                    if max_seconds is not None and time.monotonic() - start_time >= max_seconds:
                        vprint("Maximum runtime reached; stopping simulation.")
                        break
                    _one_century()
    except TimeoutError:
        vprint("Maximum runtime reached; stopping simulation.")
    finally:
        if console_ui:
            console_ui.close()
    engine.save_qtables()
    shutdown_pool()


class SimulationLoop:
    """Simplified century-by-century simulation interface."""

    def __init__(
        self,
        nations: Dict[int, Nation],
        *,
        use_neat: bool | None = None,
        use_rust_events: bool | None = None,
        qtable_path: str = str(Path.cwd() / "events_q_table.yml"),
        log_path: str | None = str(Path.cwd() / "sim_log.jsonl"),
        watch_nation: str | None = None,
    ) -> None:
        self.nations = nations
        self.engine = EventDecisionEngine(
            EVENTS,
            use_neat=use_neat,
            use_rust=use_rust_events,
            qtable_path=qtable_path,
        )
        self.century = 0
        self.watch_nation = watch_nation
        self.log_path = Path(log_path).resolve() if log_path else None
        self.recent_logs: List[tuple[str, list]] = []
        if self.log_path:
            open(self.log_path, "w").close()
        # Propagate watch to model-level wprint helper
        flags.WATCHED_NATION = watch_nation

    def save_qtables(self) -> None:
        """Persist reinforcement learning tables."""
        self.engine.save_qtables()

    def export_state(self, path: Path) -> None:
        """Write a minimal JSON snapshot of all nations."""
        import json

        data = {
            n.name: {
                "population": n.population,
                "economy": n.economy_linear,
                "economy_log": n.economy,
                "stability": n.stability,
                "cities": [c.name for c in n.cities],
            }
            for n in self.nations.values()
        }
        with open(path, "w") as fh:
            json.dump({"century": self.century, "nations": data}, fh)

    def import_state(self, path: Path) -> None:
        """Load state created by :meth:`export_state`."""
        import json

        with open(path) as fh:
            info = json.load(fh)
        self.century = int(info.get("century", 0))
        for n in self.nations.values():
            rec = info.get("nations", {}).get(n.name)
            if not rec:
                continue
            n.population = rec.get("population", n.population)
            if "economy" in rec:
                n.economy_linear = rec["economy"]
            if "economy_log" in rec:
                n.economy = rec["economy_log"]
            n.stability = rec.get("stability", n.stability)

    def step(self, human_events: list[dict] | None = None) -> None:
        """Advance the simulation by one century.

        When ``log_path`` is set, raw event data is appended to the configured
        JSONL file so runs from the GUI match CLI behaviour.
        """
        if human_events is None:
            human_events = []
        self.century += 1
        if self.century % 20 == 0:
            _apply_great_filter(self.nations)
        for n in self.nations.values():
            n.step_meta(self.century)

        century_events: list = []

        for fifth in range(1, 6):
            matrix, _, index_map = build_alliance_matrix(self.nations)

            def _issue_war(n: Nation) -> None:
                row = None
                if index_map:
                    idx = index_map.get(n.id)
                    row = matrix[idx] if idx is not None else None
                wage_war(n, self.nations, alliance_row=row)

            pooled_map(_issue_war, self.nations.values(), mode="thread")
            fifth_events: list = []
            nation_list = list(self.nations.values())
            pooled_map(
                lambda n: n.process_turn(self.nations),
                nation_list,
                mode="thread",
            )
            for n in nation_list:
                _maybe_apply_soft_great_filter(n)
            descs = self.engine.maybe_trigger_event_batch(
                nation_list, collect=self.log_path is not None
            )
            for ev in list(human_events):
                try:
                    nn = next(x for x in nation_list if x.name == ev.get("nation"))
                except StopIteration:
                    continue
                res = self.engine.force_event(
                    nn,
                    ev.get("event", "plague"),
                    int(ev.get("choice", 0)),
                    collect=self.log_path is not None,
                )
                human_events.remove(ev)
                if res is not None:
                    if self.log_path:
                        assert isinstance(res, dict)
                        fifth_events.append({"nation": nn.name, **res, "manual": True})
            pooled_map(lambda n: n.finalize_turn(), nation_list, mode="thread")
            for nid, desc in zip([n.id for n in nation_list], descs):
                if desc is not None:
                    if self.log_path:
                        assert isinstance(desc, dict)
                        fifth_events.append({"nation": self.nations[nid].name, **desc})
                    else:
                        fifth_events.append({"nation": self.nations[nid].name, "event": str(desc)})
            _handle_internal_conflicts(self.nations, self.century)
            process_planets()
            update_star_ownership(self.nations)
            update_border_pressures(self.nations)

            century_events.extend(fifth_events)

            if self.log_path:
                import json

                label = f"Century {self.century} F{fifth}"
                self.recent_logs.append((label, fifth_events))
                with open(self.log_path, "a") as fh:
                    json.dump({"century": self.century, "fifth": fifth, "events": fifth_events}, fh)
                    fh.write("\n")
                if len(self.recent_logs) > 20:
                    self.recent_logs.pop(0)

        # One compact summary per century
        _print_century_summary(
            self.nations, self.century, 5,
            watch=self.watch_nation,
            events=century_events,
        )
        # Update shared RNN base models from accumulated replay buffers.
        _rnn_step_base_models()
