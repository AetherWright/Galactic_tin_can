"""Century summary reporting for the console."""

from __future__ import annotations

import math
from typing import Dict, List, TYPE_CHECKING

from ..core import flags
from ..core.flags import _compact_pop
from ..galaxy.star import STARS

if TYPE_CHECKING:  # pragma: no cover
    from ..nations import Nation


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
    if not flags.VERBOSE:
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

