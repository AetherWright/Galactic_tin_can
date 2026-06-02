"""Core simulation loop for WorldSim."""

import math
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple, Set

from .ai import DiplomacyAI, DomesticPolicyAI, WarAI
from .torch_rnn import step_all_base_models as _rnn_step_base_models
from .torch_rnn import sync_all_meta_ga      as _rnn_sync_meta_ga
from .culture import Culture
from .events import EventDecisionEngine, EVENTS
from .initialization import (
    assign_initial_cities,
    add_star_systems,
    get_next_nation_name,
    init_world,
)
from .mp_utils import (
    pooled_map,
    shutdown_pool,
    planet_turn_worker,
    star_owner_worker,
)
from .models import AllianceBloc, Nation, Star, STARS
from .models.war import wage_war
from .models.diplomacy import update_war_score, accumulate_war_exhaustion, check_war_victory, enforce_victory
from .planets import PLANETS, Planet, City, Colony
from .representations import build_alliance_matrix
from .utils import APPROXIMATE, VERBOSE, gather_batches, time_limit, vprint, wprint, _compact_pop
import worldsim.utils as _wutils


# Rough ceilings used to detect runaway nations before numeric overflow.
# When a nation exceeds any threshold it is gently scaled back rather than
# waiting for the periodic great filter.
SOFT_FILTER_POP: int = 10_000_000_000
SOFT_FILTER_ECON: float = 1_000_000_000_000.0
SOFT_FILTER_MIL: float = 1_000_000.0


def _nation_row(n: "Nation", watch: "str | None" = None) -> str:
    """Return one compact status line for ``n``."""
    marker = "*" if (watch and n.name == watch) else " "
    pop   = _compact_pop(n.population)
    wars  = ",".join(str(e) for e in n.at_war) if n.at_war else "-"
    econ_s  = f"{n.economy:.1f}"
    stab_s  = f"{n.stability:.0f}"
    tech_s  = f"{n.technology.overall:.1f}"
    return (
        f"  {marker}{n.name:<16} pop={pop:<6} econ={econ_s:<5} "
        f"st={stab_s:<3} tech={tech_s:<5} "
        f"c={len(n.cities)} d={len(n.divisions)} fl={len(n.fleets)} "
        f"| {n.doctrine_signal[:12]:<12} war:[{wars}]"
    )


def _star_summary(nations: Dict[int, "Nation"]) -> str:
    """Return a single-line star ownership summary."""
    counts: Dict[str, int] = {}
    for star in STARS.values():
        owner = nations[star.owner].name if (star.owner is not None and star.owner in nations) else "None"
        counts[owner] = counts.get(owner, 0) + 1
    if not counts:
        return ""
    parts = "  ".join(f"{k}:{v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))
    return f"  Stars: {parts}"


def _watched_detail(n: "Nation") -> List[str]:
    """Return verbose detail lines for the watched nation."""
    lines: List[str] = [f"  {'─'*20} {n.name} (watched) {'─'*20}"]
    # Cities
    city_parts = "  ".join(
        f"{c.coords}=>{_compact_pop(c.population)}/inf{c.infrastructure:.0f}"
        for c in n.cities[:8]
    )
    lines.append(f"  cities:  {city_parts or '—'}")
    # Divisions
    div_parts = "  ".join(
        f"{d.template}[{d.soldiers}@{d.order}]" for d in n.divisions[:6]
    )
    lines.append(f"  divs:    {div_parts or '—'}")
    # Fleets
    for fl in n.fleets[:4]:
        ship_str = " ".join(f"{v}x{k}" for k, v in fl.ships.items())
        dest = f"→{fl.destination}({fl.travel_turns_remaining}t)" if fl.mission == "moving" else ""
        lines.append(f"  fleet#{fl.fleet_id}: [{ship_str}] {fl.state} {dest}")
    # Tech / projects
    tech_names = list(n.tech_tree.unlocked)[-5:]
    lines.append(f"  tech:    {', '.join(tech_names) or '—'}")
    if n.projects:
        proj_str = "  ".join(f"{p.name}({p.progress/p.cost*100:.0f}%)" for p in n.projects)
        lines.append(f"  projects:{proj_str}")
    # Doctrine / signal
    lines.append(
        f"  doctrine:{n.doctrine}  signal:{n.doctrine_signal}  "
        f"nuclear:{n.nuclear_stockpile}  fleets:{len(n.fleets)}"
    )
    return lines


def _print_century_summary(
    nations: Dict[int, "Nation"],
    century: int,
    fifth: int,
    watch: "str | None" = None,
    events: "list | None" = None,
    console_ui: object = None,
) -> None:
    """Emit one compact block summarising the century state."""
    if not VERBOSE:
        return

    total_pop   = sum(n.population    for n in nations.values())
    total_econ  = math.log1p(sum(n.economy_linear for n in nations.values()))
    total_wars  = sum(len(n.at_war)   for n in nations.values()) // 2
    total_fl    = sum(len(n.fleets)   for n in nations.values())

    header = (
        f"\n{'═'*60}\n"
        f" Century {century}  │  "
        f"pop={_compact_pop(total_pop)}  econ={total_econ:.1f}  "
        f"wars={total_wars}  fleets={total_fl}"
    )
    print(header)

    for n in nations.values():
        row = _nation_row(n, watch)
        print(row)
        if console_ui:
            try:
                console_ui.append_nation_log(  # type: ignore[attr-defined]
                    n.name, f"pop={_compact_pop(n.population)} econ={n.economy:.1f} st={n.stability:.0f}"
                )
            except Exception:
                pass

    # Star ownership summary
    star_line = _star_summary(nations)
    if star_line:
        print(star_line)

    # Events (notable only)
    if events:
        for ev in events:
            print(f"  [event] {ev.get('nation','?')}: {ev.get('event','?')}: {ev.get('choice','?')}")

    # Watched nation detail
    if watch:
        for n in nations.values():
            if n.name == watch:
                for line in _watched_detail(n):
                    print(line)
                break

    if console_ui:
        try:
            console_ui.update_world([  # type: ignore[attr-defined]
                f"Century {century}",
                f"pop={_compact_pop(total_pop)} econ={total_econ:.1f} wars={total_wars}",
            ])
        except Exception:
            pass




def process_planets() -> None:
    """Process planet turns in manageable chunks."""
    # TODO: incorporate seasonal effects and planetary climates
    planets = list(PLANETS.values())
    if not planets:
        return

    data = [(p.name, p.plague_level, p.radiation_level) for p in planets]
    results = pooled_map(planet_turn_worker, data, mode="process", chunksize=0)
    for name, plague, radiation in results:
        planet = PLANETS.get(name)
        if planet is not None:
            planet.plague_level = plague
            planet.radiation_level = radiation


def update_star_ownership(nations: Dict[int, Nation]) -> None:
    """Update star ownership in chunks."""
    # TODO: account for fleets when claiming stars
    stars = list(STARS.values())
    if not stars:
        return

    info = []
    economies = {nid: n.economy_linear for nid, n in nations.items()}
    militaries = {nid: n.military for nid, n in nations.items()}
    for star in stars:
        planets_data = []
        for pname in star.planet_names:
            planet = PLANETS.get(pname)
            if not planet:
                continue
            cities = [(c.owner, c.population) for c in planet.cities.values()]
            planets_data.append(cities)
        info.append((star.name, planets_data, economies, militaries))

    results = pooled_map(star_owner_worker, info, mode="process", chunksize=0)
    for name, owner in results:
        star = STARS.get(name)
        if star is not None:
            star.owner = owner


def compute_alliance_blocs(
    nations: Dict[int, Nation],
) -> Tuple[List[AllianceBloc], Dict[int, AllianceBloc]]:
    """Return blocs formed by alliances, trade, and relations.

    Uses a single BFS implementation regardless of the APPROXIMATE flag.
    Alliance IDs that are not in ``nations`` are skipped at push-time so the
    stack stays small and no validity check is needed at pop-time.
    """
    visited: Set[int] = set()
    blocs: List[AllianceBloc] = []
    bloc_map: Dict[int, AllianceBloc] = {}
    bid = 0
    for nid in nations:
        if nid in visited:
            continue
        members: Set[int] = set()
        stack = [nid]
        while stack:
            cur = stack.pop()
            if cur in members:
                continue
            members.add(cur)
            n = nations[cur]
            neighbors = set(n.alliances)
            neighbors.update(n.trade_partners)
            neighbors.update(
                pid for pid, rel in n.relations.items() if rel == "ally"
            )
            for ally in neighbors:
                if ally in nations and ally not in members:
                    stack.append(ally)
        visited.update(members)
        if len(members) <= 1:
            continue
        bloc = AllianceBloc.from_members(bid, members, nations)
        blocs.append(bloc)
        for m in members:
            bloc_map[m] = bloc
        bid += 1
    return blocs, bloc_map


def update_border_pressures(nations: Dict[int, Nation]) -> None:
    """Recalculate border pressure for all nations."""
    _, bloc_map = compute_alliance_blocs(nations)
    nation_list = list(nations.values())
    if nation_list:
        pooled_map(
            lambda n: n.update_border_pressure(nations, bloc_map),
            nation_list,
            mode="thread",
        )


def _spawn_rebel_nation(parent: Nation, nations: Dict[int, Nation], year: int) -> None:
    """Create a breakaway nation from *parent* during a civil war."""

    new_id = max(nations) + 1
    new_name = get_next_nation_name()
    all_ids = list(parent.all_ids) + [new_id]
    rebel = Nation(
        new_name,
        new_id,
        parent.culture.copy(),
        all_ids,
        ai_table_dir=parent.ai_table_dir,
    )
    nations[new_id] = rebel
    rebel.year_born = year
    rebel.last_collapse = year
    for key, ga in parent.reward_ga.items():
        rebel.reward_ga[key] = ga.clone()
        rebel.reward_ga[key].mutate(0.15)

    for other in nations.values():
        if other.id == new_id:
            continue
        if new_id not in other.all_ids:
            other.all_ids.append(new_id)
            if other.military_ai:
                other.military_ai.set_allies_dimension(len(other.all_ids))
        other.relations[new_id] = "neutral"
        other.border_pressure[new_id] = 0.0
        rebel.relations[other.id] = "neutral"
        rebel.border_pressure[other.id] = 0.0

    if parent.cities:
        primary = parent.dominant_idea()
        taken: List[City] = []
        if primary is not None:
            for c in list(parent.cities):
                planet = PLANETS.get(c.planet)
                county = planet.get_county(c.x, c.y) if planet else None
                if county and county.dominant_idea() and county.dominant_idea() != primary:
                    taken.append(c)
        if not taken:
            rebel_share = max(1, len(parent.cities) // 2)
            taken = random.sample(parent.cities, rebel_share)
        for c in taken:
            c.owner = new_id
            parent.cities.remove(c)
            rebel.cities.append(c)
            rebel.planet = c.planet
            planet = PLANETS.get(c.planet)
            county = planet.get_county(c.x, c.y) if planet else None
            if county:
                county.owner = new_id

    ratio = 0.4
    rebel.military = parent.military * ratio
    parent.military *= 1 - ratio
    div_move = int(len(parent.divisions) * ratio)
    for _ in range(div_move):
        if not parent.divisions:
            break
        rebel.divisions.append(parent.divisions.pop())

    rebel.stability = 50.0
    parent.stability = max(parent.stability, 35.0)
    parent.at_war.add(new_id)
    rebel.at_war.add(parent.id)
    rebel.economy_linear = parent.economy_linear * ratio
    vprint(f"  {parent.name} erupts in civil war! {new_name} rebels.")


def _handle_internal_conflicts(nations: Dict[int, Nation], year: int) -> None:
    """Process collapses and potential civil wars."""

    collapsed = [n.id for n in nations.values() if n.population <= 0]
    for nid in collapsed:
        old = nations[nid]
        killer_alive = (
            old.killer_id is not None
            and old.killer_id in nations
            and nations[old.killer_id].population > 0
        )
        old.evolve_meta()
        for city in old.cities:
            city.owner = None
            city.population = 0
        old.cities.clear()

        if killer_alive:
            for other in nations.values():
                if other.id == nid:
                    continue
                other.alliances.discard(nid)
                other.trade_partners.discard(nid)
                other.at_war.discard(nid)
                other.relations.pop(nid, None)
                other.border_pressure.pop(nid, None)
            del nations[nid]
            vprint(f"  {old.name} has been eliminated!")
            continue

        new_name = get_next_nation_name()
        culture = Culture.random()
        new_nation = Nation(
            new_name,
            nid,
            culture,
            old.all_ids,
            ai_table_dir=old.ai_table_dir,
        )
        new_nation.year_born = year
        new_nation.last_collapse = year
        new_nation.reward_ga = {k: ga.clone() for k, ga in old.reward_ga.items()}
        for ga in new_nation.reward_ga.values():
            ga.mutate(0.05)
        nations[nid] = new_nation
        assign_initial_cities({nid: new_nation})

        for other in nations.values():
            if other.id == nid:
                continue
            other.alliances.discard(nid)
            other.trade_partners.discard(nid)
            other.at_war.discard(nid)
            other.relations[nid] = "neutral"
        vprint(f"  {old.name} has collapsed! {new_name} rises.")

    for nation in list(nations.values()):
        if nation.population <= 0 or len(nation.cities) < 2:
            continue
        risk = 0.0
        if nation.stability < 50:
            risk = (50 - nation.stability) / 50
            risk += 0.001 * min(len(nation.cities), 5)
            risk -= min(0.3, nation.military / 50)
            risk = max(risk, 0.0)
        # Nations enjoy a 10-year grace period after collapsing
        years_since = year - nation.last_collapse
        if years_since < 10:
            risk *= years_since / 10
        if risk > 0 and random.random() < risk:
            _spawn_rebel_nation(nation, nations, year)


def _apply_soft_great_filter(nation: Nation) -> None:
    """Gently rein in a nation that has grown too powerful."""

    nation.population = max(1000, int(nation.population * 0.85))
    nation.military *= 0.85
    nation.economy_linear *= 0.85
    nation.stability = max(10.0, nation.stability * 0.9)


def _maybe_apply_soft_great_filter(nation: Nation) -> None:
    """Apply the soft great filter if *nation* exceeds overflow thresholds."""

    if (
        nation.population > SOFT_FILTER_POP
        or nation.economy_linear > SOFT_FILTER_ECON
        or nation.military > SOFT_FILTER_MIL
    ):
        _apply_soft_great_filter(nation)

def _cull_zombie_nations(nations: Dict[int, Nation], max_nations: int = 50) -> None:
    """Remove micro-states that are too small to be meaningful."""
    # First pass — instant kill genuinely dead nations
    for nid, n in list(nations.items()):
        if n.population < 2000 and len(n.cities) == 0:
            del nations[nid]
    
    # Second pass — if still over cap, absorb weakest into strongest neighbor
    if len(nations) <= max_nations:
        return
    
    sorted_nations = sorted(nations.values(), key=lambda n: n.population)
    while len(nations) > max_nations and sorted_nations:
        weakest = sorted_nations.pop(0)
        if weakest.id not in nations:
            continue
        # Find strongest neighbor by military
        absorber = max(
            (n for n in nations.values() if n.id != weakest.id),
            key=lambda n: n.military,
            default=None,
        )
        if absorber is None:
            break
        # Transfer cities and population
        for city in weakest.cities:
            city.owner = absorber.id
            absorber.cities.append(city)
        absorber.population += weakest.population
        # Clean up relations
        for n in nations.values():
            n.alliances.discard(weakest.id)
            n.trade_partners.discard(weakest.id)
            n.at_war.discard(weakest.id)
            n.relations.pop(weakest.id, None)
            n.border_pressure.pop(weakest.id, None)
        del nations[weakest.id]

def _apply_great_filter(nations: Dict[int, Nation]) -> None:
    """Severe calamity occurring every 20 centuries.

    Each nation suffers heavy losses and a chance of outright collapse.
    Nations can mitigate both via high stability or dedicated resilience
    programs, letting thoughtful play survive the filter.
    """

    for n in nations.values():
        resilience = getattr(n, "resilience", 0.0) / 100.0
        if n.population < 2000:
            n.population = 0
            continue
        stability = n.stability / 100.0
        collapse_chance = 0.1 * (1 - resilience) * (1 - stability)
        collapse_chance = max(0.01, collapse_chance)
        if random.random() < collapse_chance:
            n.population = 0
            continue
        keep = 0.5 + 0.3 * resilience + 0.1 * stability
        keep = min(0.95, keep)
        n.population = max(1000, int(n.population * keep))
        n.military *= keep
        n.economy_linear *= keep
        n.stability = max(5.0, n.stability * (0.7 + 0.2 * resilience))


def run_simulation(
    nations: Dict[int, Nation],
    turns: int | list[int] | None = 50,
    *,
    max_seconds: float | None = None,
    per_turn_timeout: bool = False,
    use_neat: bool | None = None,
    log_path: str | None = str(Path.cwd() / "sim_log.jsonl"),
    use_rust_events: bool | None = None,
    use_java_planets: bool | None = None,
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

    if use_java_planets is not None:
        from .planets import set_use_java
        set_use_java(use_java_planets)

    # Propagate watch to model-level wprint helper
    _wutils.WATCHED_NATION = watch_nation

    # Cache supporting divisions per war pair — only recalculate when 
    # divisions actually move, not every fifth
    _support_cache: Dict[Tuple[int,int], Tuple[List, List]] = {}

    def _run_war() -> None:
        matrix, _, index_map = build_alliance_matrix(nations)
        
        # Build set of active war pairs to avoid processing both directions
        processed: Set[Tuple[int,int]] = set()
        
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
        from .console_ui import ConsoleUI
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
            score = 0
            # Economy health
            if n.economy < 10.0:
                score -= 2
            if n.economy > 10.0:
                score += 2
            # Stability
            if n.stability > 75.0:
                score += 20
            if n.stability > 90.0:
                score += 20
            if n.stability > 50.0:
                score += 20
            if n.stability < 50.0:
                score -= 20
            if n.stability < 40.0:
                score -= 20
            if n.stability < 30.0:
                score -= 20
            if n.stability < 20.0:
                score -= 20
            if n.stability < 10.0:
                score -= 20
            # Star expansion
            if n.star_count >= 1:
                for _ in range(n.star_count):
                    score += 3
            # Fleet presence
            if n.fleets:
                for _ in range(len(n.fleets)):
                    score += 4
            # Military balance - has military but not at expense of economy
            if n.military > 5.0 and n.economy > 5.0:
                score += 1
            if n.military < 5 and n.economy > 5.0:
               score -= 10
               if n.divisions:
                  for _ in range(len(n.divisions)):
                       score += -4
            for ga in n.reward_ga.values():
                ga.step(score)
        for n in nations.values():
            n.step_meta(century)
        # Push MetaGA fitness into every RNN controller so epsilon and
        # reward-scale reflect the current genome's performance before the
        # next set of simulation fifths begins.
        _rnn_sync_meta_ga(nations)

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
        _wutils.WATCHED_NATION = watch_nation

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
        # Per-century MetaGA scoring (mirrors _one_century in run_simulation)
        for n in self.nations.values():
            score = 0
            if n.economy < 10.0:  score -= 2
            if n.economy > 10.0:  score += 2
            if n.stability > 75.0: score += 20
            if n.stability > 90.0: score += 20
            if n.stability > 50.0: score += 20
            if n.stability < 50.0: score -= 20
            if n.stability < 40.0: score -= 20
            if n.stability < 30.0: score -= 20
            if n.stability < 20.0: score -= 20
            if n.stability < 10.0: score -= 20
            for _ in range(n.star_count):   score += 3
            for _ in range(len(n.fleets)):  score += 4
            if n.military > 5.0 and n.economy > 5.0: score += 1
            if n.military < 5.0 and n.economy > 5.0:
                score -= 10
                score -= 4 * len(n.divisions)
            for ga in n.reward_ga.values():
                ga.step(score)
        for n in self.nations.values():
            n.step_meta(self.century)
        _rnn_sync_meta_ga(self.nations)

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

