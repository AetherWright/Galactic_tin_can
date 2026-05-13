from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple
import random

__all__ = [
    'School',
    'PowerPlant',
    'ResearchLab',
    'MilitaryBase',
    'Mine',
    'Port',
    'Factory',
    'Hospital',
    'Shipyard',
    'Spaceport',
    'NuclearFacility',
    'OrbitalDefense',
]

@dataclass
class School:
    """Improves science output."""

    x: int
    y: int
    planet: str
    owner: Optional[int] = None
    education: float = 1.0

    @property
    def coords(self) -> Tuple[int, int]:
        return self.x, self.y

    def upgrade(self, amount: float = 0.1) -> None:
        self.education += amount


@dataclass
class PowerPlant:
    """Provides energy to boost industry."""

    x: int
    y: int
    planet: str
    owner: Optional[int] = None
    output: float = field(default_factory=lambda: random.uniform(5.0, 15.0))

    @property
    def coords(self) -> Tuple[int, int]:
        return self.x, self.y

    def upgrade(self, amount: float = 3.0) -> None:
        self.output += amount


@dataclass
class ResearchLab:
    """Facility generating research output to advance technology."""

    x: int
    y: int
    planet: str
    owner: Optional[int] = None
    output: float = field(default_factory=lambda: random.uniform(5.0, 15.0))

    @property
    def coords(self) -> Tuple[int, int]:
        return self.x, self.y

    def upgrade(self, amount: float = 1.0) -> None:
        self.output += amount


@dataclass
class MilitaryBase:
    """Simple representation of a military base."""

    x: int
    y: int
    planet: str
    owner: Optional[int] = None
    defense: float = 10.0

    @property
    def coords(self) -> Tuple[int, int]:
        return self.x, self.y

    def upgrade(self, amount: float = 5.0) -> None:
        """Increase base defense by ``amount``."""
        self.defense += amount


@dataclass
class Mine:
    """Resource extraction site."""

    x: int
    y: int
    planet: str
    owner: Optional[int] = None
    output: float = field(default_factory=lambda: random.uniform(5.0, 15.0))
    uranium: float = field(default_factory=lambda: random.uniform(1.0, 5.0))

    @property
    def coords(self) -> Tuple[int, int]:
        return self.x, self.y

    def upgrade(self, amount: float = 2.0) -> None:
        """Boost mine output by ``amount`` for both metals and uranium."""
        self.output += amount
        self.uranium += amount * 0.5


@dataclass
class Port:
    """Trade port providing economic bonuses."""

    x: int
    y: int
    planet: str
    owner: Optional[int] = None
    bonus: float = field(default_factory=lambda: random.uniform(0.05, 0.2))

    @property
    def coords(self) -> Tuple[int, int]:
        return self.x, self.y

    def upgrade(self, amount: float = 0.05) -> None:
        """Increase trade bonus by ``amount``."""
        self.bonus += amount


@dataclass
class Factory:
    """Industrial building boosting economy."""

    x: int
    y: int
    planet: str
    owner: Optional[int] = None
    output: float = field(default_factory=lambda: random.uniform(10.0, 20.0))

    @property
    def coords(self) -> Tuple[int, int]:
        return self.x, self.y

    def upgrade(self, amount: float = 5.0) -> None:
        self.output += amount


@dataclass
class Hospital:
    """Provides plague resistance bonuses."""

    x: int
    y: int
    planet: str
    owner: Optional[int] = None
    level: float = 1.0

    @property
    def coords(self) -> Tuple[int, int]:
        return self.x, self.y

    def upgrade(self, amount: float = 0.5) -> None:
        self.level += amount


@dataclass
class Shipyard:
    """Improves naval capabilities."""

    x: int
    y: int
    planet: str
    owner: Optional[int] = None
    bonus: float = 0.1

    @property
    def coords(self) -> Tuple[int, int]:
        return self.x, self.y

    def upgrade(self, amount: float = 0.05) -> None:
        self.bonus += amount


@dataclass
class Spaceport:
    """Enables travel to other planets for colonisation."""

    x: int
    y: int
    planet: str
    owner: Optional[int] = None
    capacity: float = 1.0

    @property
    def coords(self) -> Tuple[int, int]:
        return self.x, self.y

    def upgrade(self, amount: float = 0.1) -> None:
        self.capacity += amount


@dataclass
class NuclearFacility:
    """Produces nuclear weapons for its owner."""

    x: int
    y: int
    planet: str
    owner: Optional[int] = None
    rate: float = field(default_factory=lambda: random.uniform(0.8, 1.2))

    @property
    def coords(self) -> Tuple[int, int]:
        return self.x, self.y

    def upgrade(self, amount: float = 0.1) -> None:
        self.rate += amount
@dataclass
class OrbitalDefense:
    """Station in orbit used to defend against strikes."""

    x: int
    y: int
    planet: str
    owner: Optional[int] = None
    strength: float = 0.1

    @property
    def coords(self) -> Tuple[int, int]:
        return self.x, self.y

    def upgrade(self, amount: float = 0.05) -> None:
        self.strength += amount
