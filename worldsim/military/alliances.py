"""Military alliance blocs — allied nations acting as one entity in war."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..nations.nation import Nation


@dataclass(slots=True)
class AllianceBloc:
    """Group of allied nations treated as a single entity."""

    id: int
    members: Set[int]
    centroid: Tuple[float, float, float]
    military: float

    @classmethod
    def from_members(
        cls, bloc_id: int, members: Set[int], nations: Dict[int, "Nation"]
    ) -> "AllianceBloc":
        x = y = z = 0.0
        m = 0.0
        for nid in members:
            n = nations[nid]
            wx, wy, wz = n.world_centroid
            x += wx
            y += wy
            z += wz
            m += n.military
        if members:
            c = (x / len(members), y / len(members), z / len(members))
        else:
            c = (0.0, 0.0, 0.0)
        return cls(bloc_id, members, c, m)
