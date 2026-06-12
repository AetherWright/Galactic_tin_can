"""Inter-nation communication: messages, range checks and supply runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Set, TYPE_CHECKING

from ..core import distance
from .peace import _should_accept_peace, make_peace

if TYPE_CHECKING:
    from ..nations.nation import Nation


#: Maximum star-distance that allows direct communication.
COMM_RANGE_LY: float = 15.0


# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------

@dataclass
class Message:
    """Communication packet exchanged between nations.

    Parameters
    ----------
    sender:
        Nation id of the sender.
    text:
        Human-readable description.
    kind:
        Semantic label used by AI to trigger behaviour.
    payload:
        Optional data for ``kind``-specific actions.
    """

    sender: int
    text: str
    kind: str = "text"
    payload: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Alliance helpers
# ---------------------------------------------------------------------------

def get_all_allies(nation: "Nation", nations: Dict[int, "Nation"]) -> Set[int]:
    """Return the transitive closure of all allied nation ids (including self)."""
    visited: Set[int] = set()
    stack = [nation.id]
    while stack:
        cur = stack.pop()
        if cur in visited or cur not in nations:
            continue
        visited.add(cur)
        for ally in nations[cur].alliances:
            if ally not in visited:
                stack.append(ally)
    return visited


# ---------------------------------------------------------------------------
# Communication helpers
# ---------------------------------------------------------------------------

def can_communicate(nation: "Nation", other: "Nation") -> bool:
    """Return ``True`` if the nations are within communication range."""
    from ..galaxy.star import STARS

    my_stars    = [s for s in STARS.values() if s.owner == nation.id]
    other_stars = [s for s in STARS.values() if s.owner == other.id]
    if my_stars and other_stars:
        edge_dist = min(
            s1.distance_to(s2) for s1 in my_stars for s2 in other_stars
        )
        return edge_dist <= COMM_RANGE_LY
    return distance(nation.world_centroid, other.world_centroid) <= COMM_RANGE_LY


def send_message(
    nation: "Nation",
    other: "Nation",
    text: str,
    *,
    kind: str = "text",
    payload: Optional[Dict[str, Any]] = None,
) -> None:
    """Append a :class:`Message` to *other*'s inbox if in range."""
    if can_communicate(nation, other):
        other.inbox.append(Message(nation.id, text, kind, payload))


def process_messages(
    nation: "Nation",
    nations: Dict[int, "Nation"],
) -> None:
    """React to queued messages, then clear the inbox."""
    for msg in list(nation.inbox):
        nation.stability = min(100.0, nation.stability + 0.5)
        if msg.kind == "supply" and msg.payload:
            nation.economy_linear += msg.payload.get("economy", 0.0)
            nation.military      += msg.payload.get("military", 0.0)
        elif msg.kind == "support_request":
            sender = nations.get(msg.sender)
            if sender:
                econ = nation.economy_linear * 0.1
                mil  = nation.military * 0.1
                nation.economy_linear -= econ
                nation.military       -= mil
                send_message(
                    nation,
                    sender,
                    "Supply shipment",
                    kind    = "supply",
                    payload = {"economy": econ, "military": mil},
                )
        elif msg.kind == "peace_offer":
            # Accept automatically when exhausted or severely weakened
            sender = nations.get(msg.sender)
            if sender and _should_accept_peace(nation, sender.id):
                make_peace(nation, sender)
    nation.inbox.clear()


def supply_nation(
    nation: "Nation",
    other: "Nation",
    *,
    military: float = 0.0,
    economy: float = 0.0,
) -> None:
    """Transfer resources to *other* without entering a war."""
    nation.military      = max(0.0, nation.military - military)
    nation.economy_linear = max(0.0, nation.economy_linear - economy)
    send_message(
        nation,
        other,
        "Supply shipment",
        kind    = "supply",
        payload = {"military": military, "economy": economy},
    )

