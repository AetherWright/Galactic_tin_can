"""Alliance bloc computation and border-pressure updates."""

from __future__ import annotations

from typing import Dict, List, Set, Tuple, TYPE_CHECKING

from ..core.parallel import pooled_map
from ..military.alliances import AllianceBloc

if TYPE_CHECKING:  # pragma: no cover
    from ..nations import Nation


def compute_alliance_blocs(
    nations: Dict[int, "Nation"],
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


def update_border_pressures(nations: Dict[int, "Nation"]) -> None:
    """Recalculate border pressure for all nations."""
    _, bloc_map = compute_alliance_blocs(nations)
    nation_list = list(nations.values())
    if nation_list:
        pooled_map(
            lambda n: n.update_border_pressure(nations, bloc_map),
            nation_list,
            mode="thread",
        )
