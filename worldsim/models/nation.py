"""Core data models used in the simulation."""

from dataclasses import dataclass, field
from typing import Dict, Set, List, Tuple, Optional, Any
from pathlib import Path
import random
import asyncio
import math

from ..ai import DomesticPolicyAI, ProjectAI, DiplomacyAI, ResearchAI, WarAI
from .diplomacy import (
    Message,
    can_communicate,
    send_message           as _diplomacy_send_message,
    process_messages       as _diplomacy_process_messages,
    supply_nation          as _diplomacy_supply_nation,
    update_border_pressure as _diplomacy_update_border_pressure,
    compute_trade_bonus    as _diplomacy_trade_bonus,
    collect_tributes       as _diplomacy_collect_tributes,
)
from .economy import Economy
from ..culture import Culture, ARCHETYPE_BONUSES, ArchetypeBonus
from ..meta_ga import RewardGA
from ..ideas import Idea
from ..leader import Leader, LeaderModel
from ..goals import GoalManager, Goal

from ..utils import (
    logistic_growth,
    polygon_centroid,
    polygon_area,
    distance,
    distance_sq,
    travel_time,
    APPROXIMATE,
    _np,
    vprint,
    wprint,
)
from ..config import load_list, load_json

from ..planets import (
    PLANETS,
    Planet,
    City,
    process_city_batch,
    Colony,
    process_colony_batch,
    School,
    PowerPlant,
    ResearchLab,
    MilitaryBase,
    Mine,
    Port,
    Factory,
    Hospital,
    Shipyard,
    Spaceport,
    NuclearFacility,
    OrbitalDefense,
)
from .tech import Technology, TechnologyTree, setup_default_tech_tree
from .war import Division, DivisionTemplate, build_division
from .fleet import Fleet, build_fleet_ship, process_fleet_movement, order_fleet_move
from .military_ai import DoctrineAI, issue_doctrine



# Message is defined in diplomacy.py and re-exported here for backwards
# compatibility so any code doing ``from worldsim.models.nation import Message``
# keeps working without change.
__all_message__ = ["Message"]




@dataclass(slots=True)
class Government:
    """Archetype mixing weights derived from culture proximity."""
    
    weights: Dict[str, float] = field(
        default_factory=lambda: {k: 1.0/8 for k in ARCHETYPE_BONUSES}
    )
    approval: float = 60.0
    
    def update_weights(self, culture: "Culture") -> None:
        """Recompute archetype weights from culture trait proximity."""
        from ..culture import ARCHETYPE_IDEALS
        import math
        
        distances = {}
        culture_vec = list(culture.asdict().values())
        for name, ideal in ARCHETYPE_IDEALS.items():
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(culture_vec, ideal)))
            distances[name] = dist
        
        # Inverse distance weighting
        inv = {k: 1.0 / (d + 1e-6) for k, d in distances.items()}
        total = sum(inv.values())
        self.weights = {k: v / total for k, v in inv.items()}
    
    def effective_bonus(self) -> ArchetypeBonus:
        """Return a weighted blend of all archetype bonuses."""
        result = ArchetypeBonus()
        for name, weight in self.weights.items():
            bonus = ARCHETYPE_BONUSES[name]
            result.stability_flat += bonus.stability_flat * weight
            result.military_flat += bonus.military_flat * weight
            result.diplomacy_mult *= (bonus.diplomacy_mult ** weight)
            result.trade_mult *= (bonus.trade_mult ** weight)
            result.economy_mult *= (bonus.economy_mult ** weight)
            result.science_mult *= (bonus.science_mult ** weight)
            result.plague_resist += bonus.plague_resist * weight
            result.ship_cost_mult *= (bonus.ship_cost_mult ** weight)
            result.tribute_mult *= (bonus.tribute_mult ** weight)
            result.stability_decay_at_peace += bonus.stability_decay_at_peace * weight
            result.stability_decay_no_expansion += bonus.stability_decay_no_expansion * weight
            result.stability_scale_per_star += bonus.stability_scale_per_star * weight
            result.ally_trade_bonus += bonus.ally_trade_bonus * weight
            result.civil_war_rebel_strength += (bonus.civil_war_rebel_strength - 1.0) * weight
        result.civil_war_rebel_strength = max(1.0, result.civil_war_rebel_strength)
        return result
    
    def dominant_archetype(self) -> str:
        """Return the archetype with highest weight."""
        return max(self.weights, key=lambda k: self.weights[k])
    
    def bonuses(self) -> Dict[str, float]:
        """Backwards compatible with existing bonuses() calls."""
        bonus = self.effective_bonus()
        return {
            "economy": bonus.economy_mult,
            "stability": bonus.stability_flat,
            "military": bonus.military_flat,
        }










@dataclass(slots=True)
class Star:
    """A star system linking several planets."""

    name: str
    planet_names: List[str]
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    cluster: int = 0
    owner: Optional[int] = None

    def distance_to(self, other: "Star") -> float:
        """Return Euclidean distance to ``other`` star."""

        return distance((self.x, self.y, self.z), (other.x, other.y, other.z))

    def travel_time_to(self, other: "Star") -> float:
        """Return travel time in days to ``other`` star."""

        return travel_time(self.distance_to(other), space=True)

    def update_owner(self, nations: Dict[int, "Nation"]) -> None:
        counts: Dict[int, int] = {}
        for pname in self.planet_names:
            planet = PLANETS.get(pname)
            if not planet:
                continue
            pop_by_owner: Dict[int, int] = {}
            for city in planet.cities.values():
                if city.owner is not None:
                    pop_by_owner[city.owner] = pop_by_owner.get(city.owner, 0) + city.population
            if not pop_by_owner:
                continue
            owner = max(pop_by_owner.items(), key=lambda kv: kv[1])[0]
            counts[owner] = counts.get(owner, 0) + 1
        new_owner = max(counts.items(), key=lambda kv: kv[1])[0] if counts else None
        if new_owner != self.owner:
            self.owner = new_owner


STARS: Dict[str, Star] = {}



@dataclass(slots=True)
class NationalProject:
    """Large-scale construction tracked at the nation level."""

    name: str
    cost: float
    progress: float = 0.0
    on_complete: Optional[callable] = None
    prereqs: Set[str] = field(default_factory=set)

    def advance(self, amount: float) -> bool:
        """Increase progress by ``amount`` and return ``True`` if finished."""
        remaining = max(self.cost - self.progress, 0.0)
        factor = remaining / self.cost if self.cost else 1.0
        self.progress += amount * factor
        return self.progress >= self.cost


@dataclass(slots=True)
class ProjectSpec:
    """Specification for a buildable national project."""

    cost: float
    on_complete: callable
    prereqs: Set[str] = field(default_factory=set)


PROJECT_CATALOG: Dict[str, ProjectSpec] = {
    "Highway Network": ProjectSpec(
        100.0,
        lambda n: setattr(n, "infrastructure", n.infrastructure + 20),
    ),
    "Research Complex": ProjectSpec(
        80.0,
        lambda n: setattr(
            n.technology, "science", min(100.0, n.technology.science + 10.0)
        ),
        {"Highway Network"},
    ),
    "Orbital Defense Grid": ProjectSpec(
        120.0,
        lambda n: setattr(n, "military", n.military + 20),
    ),
    "Mega Dam": ProjectSpec(
        90.0,
        lambda n: (
            setattr(n, "infrastructure", n.infrastructure + 15),
            setattr(n, "economy_linear", n.economy_linear + 20),
        ),
        {"Highway Network"},
    ),
    "AI Governance System": ProjectSpec(
        110.0,
        lambda n: (
            setattr(n, "stability", min(100.0, n.stability + 20)),
            setattr(n.technology, "industry", min(100.0, n.technology.industry + 10.0)),
        ),
        {"Research Complex"},
    ),
    "Resilience Program": ProjectSpec(
        130.0,
        lambda n: setattr(n, "resilience", min(100.0, n.resilience + 30.0)),
    ),
    "Orbital Shipyard": ProjectSpec(
        150.0,
        lambda n: (
            setattr(n, "military", n.military + 30),
            setattr(n, "infrastructure", n.infrastructure + 10),
        ),
        {"Orbital Defense Grid"},
    ),
}

# Maintains deterministic order for project indexing
PROJECT_NAMES: List[str] = list(PROJECT_CATALOG.keys())

# Resource costs for constructing various assets. Loaded from an external
# configuration file to keep this module lightweight.
RESOURCE_COSTS: Dict[str, Dict[str, float]] = load_json("resource_costs")


@dataclass(slots=True)
class Nation:
    name: str
    id: int
    culture: Culture
    all_ids: List[int]
    population: int = 1_000_000
    econ: Economy = field(default_factory=Economy)
    military: float = 50.0
    technology: Technology = field(default_factory=Technology)
    stability: float = 80.0
    resilience: float = 0.0
    infrastructure: float = 30.0
    territory: int = 100
    planet: str = "Earth"
    relations: Dict[int, str] = field(init=False)
    alliances: Set[int] = field(default_factory=set)
    trade_partners: Set[int] = field(default_factory=set)
    at_war: Set[int] = field(default_factory=set)
    cities: List[City] = field(default_factory=list)
    bases: List[MilitaryBase] = field(default_factory=list)
    mines: List[Mine] = field(default_factory=list)
    ports: List[Port] = field(default_factory=list)
    factories: List[Factory] = field(default_factory=list)
    hospitals: List[Hospital] = field(default_factory=list)
    shipyards: List[Shipyard] = field(default_factory=list)
    spaceports: List[Spaceport] = field(default_factory=list)
    schools: List[School] = field(default_factory=list)
    power_plants: List[PowerPlant] = field(default_factory=list)
    labs: List[ResearchLab] = field(default_factory=list)
    nuke_plants: List[NuclearFacility] = field(default_factory=list)
    orbital_defenses: List[OrbitalDefense] = field(default_factory=list)
    colonies: List[Colony] = field(default_factory=list)
    projects: List[NationalProject] = field(default_factory=list)
    completed_projects: List[str] = field(default_factory=list)
    border_pressure: Dict[int, float] = field(default_factory=dict)
    nuclear_stockpile: int = 0
    divisions: List[Division] = field(default_factory=list)
    division_templates: Dict[str, DivisionTemplate] = field(default_factory=dict)
    fleets: List[Fleet] = field(default_factory=list)
    # Accumulated fatigue per enemy nation id — drives peace negotiations.
    war_exhaustion: Dict[int, float] = field(default_factory=dict)
    # Running war score per enemy id (bounded ±1): positive = winning.
    war_score: Dict[int, float] = field(default_factory=dict)
    # Active war goal per enemy id (WarGoal value strings).
    war_goals: Dict[int, str] = field(default_factory=dict)
    # Tribute owed per creditor id (economy per fifth, decays 1 %/fifth).
    tribute_debts: Dict[int, float] = field(default_factory=dict)
    doctrine: str = "Balanced"
    # Military doctrine signal set by DoctrineAI each fifth (after civ/diplo AIs)
    doctrine_signal: str = "defensive"
    # Index of the civilian AI's last chosen action (read by DoctrineAI)
    last_civilian_action: int = 0
    # Post-civilian doctrine selector (optional — None disables doctrine updates)
    doctrine_ai: Optional[DoctrineAI] = field(default=None)
    available_doctrines: List[str] = field(
        default_factory=lambda: load_list("default_doctrines")
    )
    tech_tree: TechnologyTree = field(default_factory=TechnologyTree)
    tech_bonuses: Dict[str, float] = field(default_factory=dict)
    ideas: Dict[str, Idea] = field(default_factory=dict)
    government: Government = field(default_factory=Government)
    centroid: Tuple[float, float] = (0.0, 0.0)
    world_centroid: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    ai_table_dir: Optional[Path] = None
    military_ai: Optional[WarAI] = field(default=None)
    civilian_ai: Optional[DomesticPolicyAI] = field(default=None)
    project_ai: Optional[ProjectAI] = field(default=None)
    diplomacy_ai: Optional[DiplomacyAI] = field(default=None)
    research_ai: Optional[ResearchAI] = field(default=None)
    action_queue: List[int] = field(default_factory=list)
    inbox: List[Message] = field(default_factory=list)
    reward_ga: Dict[str, RewardGA] = field(default_factory=dict)
    leader_model: LeaderModel = field(default_factory=LeaderModel)
    leader: Leader = field(init=False)
    goals: GoalManager = field(default_factory=GoalManager)
    year_born: int = 0
    last_collapse: int = 0
    current_year: int = 0
    killer_id: Optional[int] = None
    event_history: Dict[str, int] = field(default_factory=dict)
    # Set to True the first time _update_centroids() runs in a given turn so
    # the redundant call at the end of process_turn() is skipped.
    _centroid_updated_this_turn: bool = field(default=False, init=False, repr=False)

    # ------------------------------------------------------------------
    # AI persistence helpers
    # ------------------------------------------------------------------
    def _ai_table_path(self, role: str) -> Optional[Path]:
        if self.ai_table_dir is None:
            return None
        return self.ai_table_dir / f"nation_{self.id}_{role}.yml"

    def _init_ai_controllers(self, ally_dim: int) -> None:
        if self.ai_table_dir is not None and not isinstance(self.ai_table_dir, Path):
            self.ai_table_dir = Path(self.ai_table_dir)
        military_path = self._ai_table_path("military")
        civilian_path = self._ai_table_path("civilian")
        project_path = self._ai_table_path("projects")
        diplomacy_path = self._ai_table_path("diplomacy")
        research_path = self._ai_table_path("research")
        self.military_ai = WarAI(allies_dim=ally_dim, table_path=military_path)
        self.civilian_ai = DomesticPolicyAI(21,n_inputs=20,hidden_layers=(32, 24, 16, 10, 8, 7, 7, 8, 10, 16, 24, 32),table_path=civilian_path,)
        self.project_ai = ProjectAI(
            len(PROJECT_CATALOG), n_inputs=6, table_path=project_path
        )
        self.diplomacy_ai = DiplomacyAI(3, n_inputs=5, table_path=diplomacy_path)
        self.research_ai = ResearchAI(
            len(self.tech_tree.nodes), n_inputs=4, table_path=research_path
        )
        # DoctrineAI runs AFTER civilian and diplomacy AIs each fifth
        doctrine_path = self._ai_table_path("doctrine")
        self.doctrine_ai = DoctrineAI(table_path=doctrine_path)

    # ------------------------------------------------------------------
    # Economy proxies
    # ------------------------------------------------------------------
    @property
    def economy(self) -> float:
        """Return the logarithmically scaled economy value."""

        return self.econ.funds

    @economy.setter
    def economy(self, value: float) -> None:
        self.econ.funds = max(0.0, value)

    @property
    def economy_linear(self) -> float:
        """Return the underlying linear economy value."""

        return self.econ.linear_funds

    @economy_linear.setter
    def economy_linear(self, value: float) -> None:
        self.econ.set_linear_funds(value)

    @property
    def resources(self) -> Dict[str, float]:
        return self.econ.resources

    @resources.setter
    def resources(self, value: Dict[str, float]) -> None:
        self.econ.resources = value

    @property
    def resource_caps(self) -> Dict[str, float]:
        return self.econ.caps

    @resource_caps.setter
    def resource_caps(self, value: Dict[str, float]) -> None:
        self.econ.caps = value

    def has_resources(self, cost: Dict[str, float]) -> bool:
        return self.econ.has_resources(cost)

    def spend_resources(self, cost: Dict[str, float]) -> None:
        self.econ.spend_resources(cost)

    def add_resource(self, kind: str, amount: float) -> None:
        self.econ.add_resource(kind, amount)

    def dominant_idea(self) -> str | None:
        """Return the most prevalent idea across the nation."""
        totals: Dict[str, int] = {}
        for idea in self.ideas.values():
            totals[idea.name] = totals.get(idea.name, 0) + idea.followers
        for city in self.cities:
            for ide in city.ideas.values():
                totals[ide.name] = totals.get(ide.name, 0) + ide.followers
            planet = PLANETS.get(city.planet)
            county = planet.get_county(city.x, city.y) if planet else None
            if county and county.owner == self.id:
                for ide in county.ideas.values():
                    totals[ide.name] = totals.get(ide.name, 0) + ide.followers
        if not totals:
            return None
        return max(totals, key=totals.get)

    def __post_init__(self) -> None:
        self.relations = {nid: "neutral" for nid in self.all_ids if nid != self.id}
        self.border_pressure = {nid: 0.0 for nid in self.all_ids if nid != self.id}
        setup_default_tech_tree(self)
        self.division_templates["Infantry"] = DivisionTemplate("Infantry", 1000)
        ally_dim = max(1, len(self.all_ids)) if self.all_ids else 1
        self._init_ai_controllers(ally_dim)
        self.doctrine = self.military_ai.create_doctrine()
        if self.doctrine not in self.available_doctrines:
            self.available_doctrines.append(self.doctrine)
        self.reward_ga = {
            "research": RewardGA(4),
            "projects": RewardGA(6),
            "civilian": RewardGA(6),
            "diplomacy": RewardGA(5),
            "events": RewardGA(5),
        }
        self.goals.spawn_goals(self, 2)
        self.leader = self.leader_model.generate(self)
        self._update_centroids()
    def _fleet_count(self):
        return float(len(self.fleets))
    def _update_centroids(self) -> None:
        """Recalculate 2-D and 3-D centroids based on owned cities."""
        if not self.cities:
            self.centroid = (0.0, 0.0)
            planet = PLANETS.get(self.planet)
            if planet:
                self.world_centroid = planet.coords
            else:
                self.world_centroid = (0.0, 0.0, 0.0)
            self._centroid_updated_this_turn = True
            return
        self.centroid = polygon_centroid([c.coords for c in self.cities])
        xs = ys = zs = 0.0
        for c in self.cities:
            wx, wy, wz = c.world_coords
            xs += wx
            ys += wy
            zs += wz
        n = len(self.cities)
        self.world_centroid = (xs / n, ys / n, zs / n)
        self._centroid_updated_this_turn = True

    def compute_reward(
        self, task: str, state: List[float], new_state: List[float]
    ) -> float:
        """Return weighted reward for *task* based on meta-learning."""
        diff = [n - s for n, s in zip(new_state, state)]
        ga = self.reward_ga.get(task)
        if ga:
            weights = ga.weights
            if len(weights) < len(diff):
                weights = weights + [1.0] * (len(diff) - len(weights))
            return sum(w * d for w, d in zip(weights, diff))
        return sum(diff)

    def step_meta(self, year: int) -> None:
        self.current_year = year
        self.leader_model.step(1)

    def evolve_meta(self) -> None:
        for ga in self.reward_ga.values():
            ga.evolve()
        self.leader_model.evolve()
        self.leader = self.leader_model.generate(self)
        self.last_collapse = self.current_year

    # ------------------------------------------------------------------
    # Idea helpers

    def culture_with_ideas(self) -> Culture:
        """Return culture modified by active ideas."""
        result = self.culture.copy()
        for idea in self.ideas.values():
            result = idea.influence(result, self.population)
        for city in self.cities:
            for idea in city.ideas.values():
                result = idea.influence(result, city.population)
        for planet in PLANETS.values():
            for county in planet.counties.values():
                if county.owner == self.id:
                    for idea in county.ideas.values():
                        result = idea.influence(result, county.rural_population)
        return result

    def culture_with_leader(self) -> Culture:
        """Return culture further adjusted by the current leader."""
        result = self.culture_with_ideas()
        if hasattr(self, "leader") and self.leader:
            for trait in result.__dataclass_fields__:
                val = getattr(result, trait)
                lval = getattr(self.leader.culture, trait)
                setattr(result, trait, (val + lval) / 2)
        return result

    # ------------------------------------------------------------------
    # Territory helpers

    @property
    def _border_radius(self) -> float:
        """Return radius of territory treated as a circle."""
        return math.sqrt(self.territory / math.pi)

    @property
    def _border_radius_sq(self) -> float:
        """Return squared border radius to avoid repeated sqrt."""
        return self.territory / math.pi

    def _within_borders(self, city: City) -> bool:
        """Return ``True`` if ``city`` lies inside current borders."""
        if not self.cities:
            return True
        dist_sq = distance_sq(self.centroid, city.coords)
        return dist_sq <= self._border_radius_sq

    def _area_with_city(self, city: City) -> float:
        coords = [c.coords for c in self.cities] + [city.coords]
        return polygon_area(coords)

    def _can_add_city(self, city: City) -> bool:
        if not self._within_borders(city):
            return False
        new_area = self._area_with_city(city)
        return new_area <= self.territory

    def get_all_allies(self, nations: Dict[int, "Nation"]) -> Set[int]:
        visited: Set[int] = set()
        stack = [self.id]
        while stack:
            cur = stack.pop()
            if cur in visited or cur not in nations:
                continue
            visited.add(cur)
            for ally in nations[cur].alliances:
                if ally not in visited:
                    stack.append(ally)
        return visited

    def can_communicate(self, other: "Nation") -> bool:
        """Return ``True`` if ``other`` is within communication range."""
        return can_communicate(self, other)

    def send_message(
        self,
        other: "Nation",
        text: str,
        *,
        kind: str = "text",
        payload: Dict[str, Any] | None = None,
    ) -> None:
        """Append a :class:`Message` to ``other``'s inbox if in range."""
        _diplomacy_send_message(self, other, text, kind=kind, payload=payload)

    def process_messages(self, nations: Dict[int, "Nation"]) -> None:
        """React to queued messages then clear the inbox."""
        _diplomacy_process_messages(self, nations)

    def supply_nation(
        self,
        other: "Nation",
        *,
        military: float = 0.0,
        economy: float = 0.0,
    ) -> None:
        """Provide resources to ``other`` without joining a war."""
        _diplomacy_supply_nation(self, other, military=military, economy=economy)

    def collect_resources(self) -> None:
        for mine in self.mines:
            planet = PLANETS.get(mine.planet)
            if not planet:
                continue
            gained_m = planet.extract_resource("metal", mine.output)
            self.add_resource("metal", gained_m)
            gained_u = planet.extract_resource("uranium", getattr(mine, "uranium", 0.0))
            self.add_resource("uranium", gained_u)
        for plant in self.power_plants:
            planet = PLANETS.get(plant.planet)
            if not planet:
                continue
            gained = planet.extract_resource("energy", plant.output)
            self.add_resource("energy", gained)
        for _city in self.cities:
            planet = PLANETS.get(_city.planet)
            if not planet:
                continue
            self.add_resource("food", planet.extract_resource("food", 5.0))

    def process_turn(self, nations: Dict[int, "Nation"]) -> None:
        self._centroid_updated_this_turn = False
        self.process_messages(nations)
        self.consider_first_strike(nations)
        self.collect_resources()
        total_pop = 0
        total_econ = 0.0
        plague_res = self.tech_bonuses.get("plague_resist", 0.0)
        gov_bonuses = self.government.bonuses()
        planet = PLANETS.get(self.planet)
        plague_level = planet.plague_level if planet else 0.0
        radiation_level = planet.radiation_level if planet else 0.0
        cbonus = self.tech_bonuses.get("city_output", 1.0)
        pop, econ = process_city_batch(
            self.cities, plague_level, radiation_level, plague_res, cbonus
        )
        total_pop += pop
        total_econ += econ
        if self.colonies:
            cpop = process_colony_batch(self.colonies, plague_level, plague_res)
            total_pop += cpop
        if _np is not None:
            if self.mines:
                arr = _np.array([m.output for m in self.mines], dtype=float)
                total_econ += float(arr.sum()) * self.tech_bonuses.get("mine_output", 1.0)
            if self.factories:
                arr = _np.array([f.output for f in self.factories], dtype=float)
                total_econ += float(arr.sum()) * self.tech_bonuses.get("factory_output", 1.0)
            if self.power_plants:
                arr = _np.array([p.output for p in self.power_plants], dtype=float)
                total_econ += float(arr.sum())
        else:
            for mine in self.mines:
                total_econ += mine.output * self.tech_bonuses.get("mine_output", 1.0)
            for fac in self.factories:
                total_econ += fac.output * self.tech_bonuses.get("factory_output", 1.0)
            for plant in self.power_plants:
                total_econ += plant.output
        research_bonus = 0.0
        if _np is not None and self.schools:
            arr = _np.array([s.education for s in self.schools], dtype=float)
            research_bonus += float(arr.sum())
        else:
            research_bonus += sum(s.education for s in self.schools)
        if _np is not None and self.labs:
            arr = _np.array([l.output for l in self.labs], dtype=float)
            research_bonus += float(arr.sum())
        else:
            research_bonus += sum(l.output for l in self.labs)

        if _np is not None and self.hospitals:
            levels = _np.array([hos.level for hos in self.hospitals], dtype=float)
            for hos, level in zip(self.hospitals, levels.tolist()):
                planet = PLANETS.get(hos.planet)
                if planet:
                    planet.plague_level = max(0.0, planet.plague_level - level * 0.005)
        else:
            for hos in self.hospitals:
                planet = PLANETS.get(hos.planet)
                if planet:
                    planet.plague_level = max(0.0, planet.plague_level - hos.level * 0.005)

        if _np is not None and self.ports:
            arr = _np.array([p.bonus for p in self.ports], dtype=float)
            port_bonus = float(arr.sum()) * self.tech_bonuses.get("port_bonus", 1.0)
        else:
            port_bonus = sum(p.bonus for p in self.ports) * self.tech_bonuses.get("port_bonus", 1.0)
        
        self.population = total_pop
        trade_bonus = _diplomacy_trade_bonus(self, nations)
        _diplomacy_collect_tributes(self, nations)
        econ_mult = self.tech_bonuses.get("economy_mult", 1.0)
        if "economy" in gov_bonuses:
            econ_mult *= gov_bonuses["economy"]
        self.economy_linear = total_econ * (1 + port_bonus) * econ_mult + trade_bonus

        self.tech_tree.research(
            self.economy * 0.05 + research_bonus,
            self,
            self.research_ai,
        )
        self.produce_nuclear_weapons()
        self.progress_projects()

        # Stability and tech adjustments
        stab_flat = gov_bonuses.get("stability", 0.0)
        self.stability = max(0.0, min(self.stability + stab_flat * 0.1 + random.uniform(-0.5, 1), 100.0))
        self.technology.advance(self.economy, research_bonus)

        self.government.approval += (self.stability - 50) / 100
        self.government.approval = max(0.0, min(self.government.approval, 100.0))
        if self.government.approval < 20 and random.random() < 0.1:
            # Low approval triggers culture drift toward a random archetype
            from ..culture import ARCHETYPE_IDEALS
            import math
            target = random.choice(list(ARCHETYPE_IDEALS.keys()))
            ideal = ARCHETYPE_IDEALS[target]
            for i, trait in enumerate(self.culture.__dataclass_fields__):
                current = getattr(self.culture, trait)
                setattr(self.culture, trait, current + (ideal[i] - current) * 0.2)
                setattr(self.culture, trait, max(0.0, min(1.0, getattr(self.culture, trait))))
            self.government.update_weights(self.culture)
            self.government.approval = 60.0
        if self.civilian_ai:
            self.process_action_queue()
            self._apply_civilian_ai()
        else:
            self._random_civilian_actions(nations)
        process_fleet_movement(self, nations)   # physics: ticks movement, handles arrivals
        issue_doctrine(self, nations)            # strategy: doctrine + fleet FSM decisions

        for div in self.divisions:
            div.experience = min(div.experience + 0.01, 2.0)

        if self.cities and not self._centroid_updated_this_turn:
            self._update_centroids()

        if self.stability < 40 and self.alliances:
            ally_id = random.choice(list(self.alliances))
            if ally_id in nations:
                self.send_message(
                    nations[ally_id],
                    "Requesting support",
                    kind="support_request",
                )

    def build_city(self) -> None:
        planet = PLANETS.get(self.planet)
        if not planet:
            return
        cost = RESOURCE_COSTS["city"]
        if not self.has_resources(cost):
            return
        free = [co for co in planet.iter_colonies() if co.owner is None]
        for co in free:
            if self._can_add_city(co):
                co.owner = self.id
                planet.register_colony_usage(co)
                city = planet.upgrade_colony(co, self.id)
                self.cities.append(city)
                if co in self.colonies:
                    self.colonies.remove(co)
                self.infrastructure += 1
                self.spend_resources(cost)
                self._update_centroids()
                break

    def build_base(self) -> None:
        planet = PLANETS.get(self.planet)
        if not planet or not self.cities:
            return
        cost = RESOURCE_COSTS["base"]
        if not self.has_resources(cost):
            return
        anchor = max(self.cities, key=lambda c: c.population)
        if (anchor.x, anchor.y) in planet.bases:
            # Existing base; treat as capacity upgrade
            self.spend_resources(cost)
            return
        base = MilitaryBase(anchor.x, anchor.y, self.planet, owner=self.id)
        planet.add_base(base)
        self.bases.append(base)
        self.spend_resources(cost)

    def build_mine(self) -> None:
        planet = PLANETS.get(self.planet)
        if not planet or not self.cities:
            return
        cost = RESOURCE_COSTS["mine"]
        if not self.has_resources(cost):
            return
        anchor = max(self.cities, key=lambda c: c.population)
        if (anchor.x, anchor.y) in planet.mines:
            # Existing mine; invest to expand output
            self.spend_resources(cost)
            return
        mine = Mine(anchor.x, anchor.y, self.planet, owner=self.id)
        planet.add_mine(mine)
        self.mines.append(mine)
        self.spend_resources(cost)

    def build_port(self) -> None:
        planet = PLANETS.get(self.planet)
        if not planet or not self.cities:
            return
        cost = RESOURCE_COSTS["port"]
        if not self.has_resources(cost):
            return
        anchor = max(self.cities, key=lambda c: c.population)
        if (anchor.x, anchor.y) in planet.ports:
            # Upgrade existing port rather than build anew
            self.spend_resources(cost)
            return
        port = Port(anchor.x, anchor.y, self.planet, owner=self.id)
        planet.add_port(port)
        self.ports.append(port)
        self.spend_resources(cost)

    def build_factory(self) -> None:
        planet = PLANETS.get(self.planet)
        if not planet or not self.cities:
            return
        cost = RESOURCE_COSTS["factory"]
        if not self.has_resources(cost):
            return
        anchor = max(self.cities, key=lambda c: c.population)
        if (anchor.x, anchor.y) in planet.factories:
            # Expand current factory capacity
            self.spend_resources(cost)
            return
        fac = Factory(anchor.x, anchor.y, self.planet, owner=self.id)
        planet.add_factory(fac)
        self.factories.append(fac)
        self.spend_resources(cost)

    def build_hospital(self) -> None:
        planet = PLANETS.get(self.planet)
        if not planet or not self.cities:
            return
        cost = RESOURCE_COSTS["hospital"]
        if not self.has_resources(cost):
            return
        anchor = max(self.cities, key=lambda c: c.population)
        if (anchor.x, anchor.y) in planet.hospitals:
            # Upgrade existing hospital facilities
            self.spend_resources(cost)
            return
        hos = Hospital(anchor.x, anchor.y, self.planet, owner=self.id)
        planet.add_hospital(hos)
        self.hospitals.append(hos)
        self.spend_resources(cost)

    def build_shipyard(self) -> None:
        planet = PLANETS.get(self.planet)
        if not planet or not self.cities:
            return
        cost = RESOURCE_COSTS["shipyard"]
        if not self.has_resources(cost):
            return
        anchor = max(self.cities, key=lambda c: c.population)
        if (anchor.x, anchor.y) in planet.shipyards:
            # Expand current shipyard
            self.spend_resources(cost)
            return
        yard = Shipyard(anchor.x, anchor.y, self.planet, owner=self.id)
        planet.add_shipyard(yard)
        self.shipyards.append(yard)
        self.spend_resources(cost)

    def build_school(self) -> None:
        planet = PLANETS.get(self.planet)
        if not planet or not self.cities:
            return
        cost = RESOURCE_COSTS["school"]
        if not self.has_resources(cost):
            return
        anchor = max(self.cities, key=lambda c: c.population)
        if (anchor.x, anchor.y) in planet.schools:
            # Improve existing school
            self.spend_resources(cost)
            return
        school = School(anchor.x, anchor.y, self.planet, owner=self.id)
        planet.add_school(school)
        self.schools.append(school)
        self.spend_resources(cost)

    def build_power_plant(self) -> None:
        planet = PLANETS.get(self.planet)
        if not planet or not self.cities:
            return
        cost = RESOURCE_COSTS["power_plant"]
        if not self.has_resources(cost):
            return
        anchor = max(self.cities, key=lambda c: c.population)
        if (anchor.x, anchor.y) in planet.power_plants:
            # Boost existing plant efficiency
            self.spend_resources(cost)
            return
        plant = PowerPlant(anchor.x, anchor.y, self.planet, owner=self.id)
        planet.add_power_plant(plant)
        self.power_plants.append(plant)
        self.spend_resources(cost)

    def build_lab(self) -> None:
        planet = PLANETS.get(self.planet)
        if not planet or not self.cities:
            return
        cost = RESOURCE_COSTS["lab"]
        if not self.has_resources(cost):
            return
        anchor = max(self.cities, key=lambda c: c.population)
        if (anchor.x, anchor.y) in planet.labs:
            # Upgrade existing research lab
            self.spend_resources(cost)
            return
        lab = ResearchLab(anchor.x, anchor.y, self.planet, owner=self.id)
        planet.add_lab(lab)
        self.labs.append(lab)
        self.spend_resources(cost)

    def build_nuke_facility(self) -> None:
        planet = PLANETS.get(self.planet)
        if not planet or not self.cities:
            return
        cost = RESOURCE_COSTS["nuke_facility"]
        if not self.has_resources(cost):
            return
        anchor = max(self.cities, key=lambda c: c.population)
        if (anchor.x, anchor.y) in planet.nuke_plants:
            # Expand current nuclear facility
            self.spend_resources(cost)
            return
        fac = NuclearFacility(anchor.x, anchor.y, self.planet, owner=self.id)
        planet.add_nuke_facility(fac)
        self.nuke_plants.append(fac)
        self.spend_resources(cost)

    def build_orbital_defense(self) -> None:
        planet = PLANETS.get(self.planet)
        if not planet or not self.cities:
            return
        cost = RESOURCE_COSTS["orbital_defense"]
        if not self.has_resources(cost):
            return
        anchor = max(self.cities, key=lambda c: c.population)
        if (anchor.x, anchor.y) in planet.orbital_defenses:
            # Reinforce existing orbital defense
            self.spend_resources(cost)
            return
        od = OrbitalDefense(anchor.x, anchor.y, self.planet, owner=self.id)
        planet.add_orbital_defense(od)
        self.orbital_defenses.append(od)
        self.spend_resources(cost)

    def build_fleet_ship(self, template_name: str = "Frigate") -> None:
        """Build one ship of ``template_name`` and add it to a fleet."""
        build_fleet_ship(self, template_name)

    def move_fleet(
        self,
        fleet: Fleet,
        destination: str,
        distance_ly: float,
        divisions_to_load: Optional[List[Division]] = None,
    ) -> bool:
        """Order ``fleet`` to move to ``destination``, optionally loading troops.

        Returns ``True`` if the order was accepted (fleet is now underway).
        """
        return order_fleet_move(self, fleet, destination, distance_ly, divisions_to_load)

    def build_spaceport(self) -> None:
        planet = PLANETS.get(self.planet)
        if not planet or not self.cities:
            return
        cost = RESOURCE_COSTS["spaceport"]
        if not self.has_resources(cost):
            return
        anchor = max(self.cities, key=lambda c: c.population)
        if (anchor.x, anchor.y) in planet.spaceports:
            # Extend existing spaceport capacity
            self.spend_resources(cost)
            return
        port = Spaceport(anchor.x, anchor.y, self.planet, owner=self.id)
        planet.add_spaceport(port)
        self.spaceports.append(port)
        self.spend_resources(cost)

    def colonize_planet(self) -> None:
        """Attempt to found a city on another planet if one is free."""
        # pick any planet with an unowned colony
        candidates = []
        for planet in PLANETS.values():
            free = [c for c in planet.iter_colonies() if c.owner is None]
            if free:
                candidates.append((planet, free))
        if not candidates:
            return
        candidates.sort(key=lambda item: -len(item[1]))
        for target, free in candidates:
            for colony in free:
                if self._can_add_city(colony):
                    colony.owner = self.id
                    target.register_colony_usage(colony)
                    city = target.upgrade_colony(colony, self.id)
                    self.cities.append(city)
                    if colony in self.colonies:
                        self.colonies.remove(colony)
                    self._update_centroids()
                    return

    def upgrade_assets(self) -> None:
        """Invest economy into upgrading owned infrastructure."""

        if self.economy_linear <= 0:
            return
        if self.cities and self.economy_linear >= 20:
            self.cities[0].upgrade()
            self.economy_linear -= 20
        if self.bases and self.economy_linear >= 15:
            self.bases[0].upgrade()
            self.economy_linear -= 15
        if self.mines and self.economy_linear >= 10:
            self.mines[0].upgrade()
            self.economy_linear -= 10
        if self.ports and self.economy_linear >= 15:
            self.ports[0].upgrade()
            self.economy_linear -= 15
        if self.factories and self.economy_linear >= 20:
            self.factories[0].upgrade()
            self.economy_linear -= 20
        if self.hospitals and self.economy_linear >= 10:
            self.hospitals[0].upgrade()
            self.economy_linear -= 10
        if self.shipyards and self.economy_linear >= 20:
            self.shipyards[0].upgrade()
            self.economy_linear -= 20
        if self.schools and self.economy_linear >= 10:
            self.schools[0].upgrade()
            self.economy_linear -= 10
        if self.labs and self.economy_linear >= 10:
            self.labs[0].upgrade()
            self.economy_linear -= 10
        if self.power_plants and self.economy_linear >= 15:
            self.power_plants[0].upgrade()
            self.economy_linear -= 15
        if self.spaceports and self.economy_linear >= 20:
            self.spaceports[0].upgrade()
            self.economy_linear -= 20

    def produce_nuclear_weapons(self) -> None:
        """Convert uranium, metal and economy into nuclear stockpile."""
        if "Atomic Engineering" not in self.tech_tree.unlocked:
            return
        uranium = self.resources.get("uranium", 0.0)
        metal = self.resources.get("metal", 0.0)
        base = int(min(uranium // 10, metal // 20, self.economy_linear // 100))
        if base <= 0:
            return
        rate = sum(f.rate for f in self.nuke_plants) if self.nuke_plants else 1.0
        possible = int(
            min(
                base * rate,
                self.resources.get("uranium", 0.0) // 10,
                self.resources.get("metal", 0.0) // 20,
                self.economy_linear // 100,
            )
        )
        if possible > 0:
            self.nuclear_stockpile += possible
            self.resources["uranium"] -= possible * 10
            self.resources["metal"] -= possible * 20
            self.economy_linear -= possible * 100

    def launch_nuclear_strike(self, enemy: "Nation") -> None:
        """Inflict heavy losses on ``enemy`` using one warhead.

        Beyond physical devastation this strike now also causes a
        diplomatic collapse between the two nations and destabilises the
        attacker. Damage to the target nation is intentionally severe to
        reflect the catastrophic nature of nuclear warfare.
        """

        # Enter open conflict and ruin diplomatic relations
        self.relations[enemy.id] = "enemy"
        enemy.relations[self.id] = "enemy"
        self.at_war.add(enemy.id)
        enemy.at_war.add(self.id)

        self.nuclear_stockpile -= 1

        tech_scale = 1.0 + self.technology.military / 100.0
        defense = sum(d.strength for d in enemy.orbital_defenses)
        damp = max(0.0, 1.0 - defense)

        casualties = 0

        if enemy.cities:
            target = max(enemy.cities, key=lambda c: c.population)
            city_loss = int(target.population * 0.8 * tech_scale * damp)
            target.population = max(0, target.population - city_loss)
            infra_loss = max(1, int(target.infrastructure * 0.7 * damp))
            target.infrastructure = max(0, target.infrastructure - infra_loss)
            casualties += city_loss
        else:
            casualties += int(enemy.population * 0.2 * damp)

        enemy.population = max(0, enemy.population - casualties)
        enemy.economy_linear *= 0.3 * damp
        enemy.stability = max(0.0, enemy.stability - 40 * damp)
        self.stability = max(0.0, self.stability - 10)

        planet = PLANETS.get(enemy.planet)
        if planet:
            planet.radiation_level = min(1.0, planet.radiation_level + 0.4 * damp)

        wprint(self.name, f"  {self.name} launches a nuclear strike on {enemy.name}!")

    def launch_first_strike(self, enemies: List["Nation"]) -> None:
        """Launch all available warheads across ``enemies``.

        Warheads are distributed in round-robin fashion. After each hit the
        targeted nation may immediately retaliate with a single strike if
        it has remaining weapons. This models a rapid escalation typical
        of all-out nuclear exchanges.
        """

        if self.nuclear_stockpile <= 0 or not enemies:
            return

        idx = 0
        # Disperse warheads sequentially among enemies
        while self.nuclear_stockpile > 0:
            enemy = enemies[idx % len(enemies)]
            self.launch_nuclear_strike(enemy)
            if enemy.nuclear_stockpile > 0 and random.random() < 0.5:
                enemy.launch_nuclear_strike(self)
            idx += 1

    def consider_first_strike(self, nations: Dict[int, "Nation"]) -> None:
        """Allow the military AI to initiate a nuclear first strike."""

        if not self.military_ai or self.nuclear_stockpile <= 0:
            return
        potential = [
            n
            for n in nations.values()
            if n.id != self.id
            and n.id not in self.at_war
            and self.relations.get(n.id, "neutral") != "ally"
        ]
        if not potential:
            return
        state = [
            float(self.nuclear_stockpile),
            self.military,
            self.economy,
            self.stability,
            float(sum(e.nuclear_stockpile for e in potential)),
        ]
        act = self.military_ai.choose_action(state)
        if act == 1:
            target = max(potential, key=lambda n: n.military)
            self.launch_first_strike([target])

    def available_projects(self) -> List[str]:
        """Return project names that can currently be started."""
        opts: List[str] = []
        for pname, spec in PROJECT_CATALOG.items():
            if pname in self.completed_projects:
                continue
            if any(p.name == pname for p in self.projects):
                continue
            if not spec.prereqs.issubset(set(self.completed_projects)):
                continue
            opts.append(pname)
        return opts

    def start_project(self, name: Optional[str] = None) -> None:
        """Begin a new national project if resources allow."""
        if len(self.projects) >= 2 or self.economy_linear < 50:
            return
        options = self.available_projects()
        if not options:
            return
        state = self._civilian_state()
        if name is None or name not in options:
            idx = self.project_ai.choose_action(state)
            choice = PROJECT_NAMES[idx % len(PROJECT_NAMES)]
            if choice not in options:
                choice = options[0]
        else:
            choice = name
        spec = PROJECT_CATALOG[choice]
        self.projects.append(
            NationalProject(choice, spec.cost, 0.0, spec.on_complete, spec.prereqs)
        )
        new_state = self._civilian_state()
        reward = self.compute_reward("projects", state, new_state)
        self.project_ai.train(state, PROJECT_NAMES.index(choice), reward, new_state)

    def progress_projects(self) -> None:
        """Spend economy to advance national projects."""
        if not self.projects or self.economy_linear <= 0:
            return
        invest = min(self.economy_linear, 20 * len(self.projects))
        per = invest / len(self.projects)
        self.economy_linear -= invest
        finished: List[NationalProject] = []
        for prj in self.projects:
            if prj.advance(per):
                if prj.on_complete:
                    prj.on_complete(self)
                self.completed_projects.append(prj.name)
                finished.append(prj)
        for prj in finished:
            if prj in self.projects:
                self.projects.remove(prj)
    @property
    def star_count(self) -> int:
        from .nation import STARS
        return sum(1 for s in STARS.values() if s.owner == self.id)

    def _civilian_state(self) -> List[float]:
        return [
            self.economy,
            self.technology.overall,
            self.military,
            self.infrastructure,
            self.stability,
            float(len(self.projects)),
            float(self.star_count),
            self._fleet_count(),           
            float(len(self.cities)),
            float(len(self.divisions)),
            float(len(self.mines)),
            float(len(self.factories)),
            float(len(self.schools)),
            float(len(self.labs)),
            float(len(self.hospitals)),
            self.resources.get("metal", 0.0) / 100.0,
            self.resources.get("uranium", 0.0) / 100.0,
            self.resources.get("energy", 0.0) / 100.0,
            float(len(self.at_war)),
            float(len(self.alliances)),
        ]
    def _execute_civilian_action(self, idx: int) -> None:
        actions = [
            self.build_city,
            self.build_base,
            self.build_mine,
            self.build_port,
            self.build_factory,
            self.build_hospital,
            self.build_shipyard,
            self.build_school,
            self.build_power_plant,
            self.build_spaceport,
            self.build_lab,
            self.build_nuke_facility,
            self.build_orbital_defense,
            lambda: build_division(self),
	    lambda: self.build_fleet_ship("Frigate"),
	    lambda: self.build_fleet_ship("Transport"),
	    lambda: self.build_fleet_ship("Cruiser"),
	    lambda: self.build_fleet_ship("Battleship"),
            self.colonize_planet,
            self.upgrade_assets,
            self.start_project,
        ]
        if 0 <= idx < len(actions):
            actions[idx]()
    def _valid_action_mask(self) -> List[bool]:
        has_nuke_tech = "Nuclear Weapons" in self.tech_tree.unlocked
        has_shipyard = len(self.shipyards) > 0
        has_spaceport = bool(self.spaceports) if hasattr(self, 'spaceports') else False
        can_colonize = self.star_count < len([s for s in STARS.values() if s.owner is None])
        
        return [
            True,           # build_city
            True,           # build_base
            True,           # build_mine
            True,           # build_port
            True,           # build_factory
            True,           # build_hospital
            True,           # build_shipyard
            True,           # build_school
            True,           # build_power_plant
            True,           # build_spaceport
            True,           # build_lab
            has_nuke_tech,  # build_nuke_facility
            True,           # build_orbital_defense
            True,           # build_division
            has_shipyard,   # frigate
            has_shipyard,   # transport
            has_shipyard,   # cruiser
            has_shipyard,   # battleship
            can_colonize,   # colonize_planet
            True,           # upgrade_assets
            True,           # start_project
        ]
    def _apply_civilian_ai(self) -> None:
        state = self._civilian_state()
        
        # Mask invalid actions before selection
        valid_mask = self._valid_action_mask()
        idx = self.civilian_ai.choose_action(state, valid_mask)
    
        self.last_civilian_action = idx
        self._execute_civilian_action(idx)
    
        # Delayed reward — evaluate against state from N turns ago
        # rather than immediate state change
        new_state = self._civilian_state()
        reward = self.compute_reward("civilian", state, new_state)
        self.civilian_ai.train(state, idx, reward, new_state)

    def process_action_queue(self, limit: int = 2) -> None:
        for _ in range(min(limit, len(self.action_queue))):
            idx = self.action_queue.pop(0)
            state = self._civilian_state()
            self._execute_civilian_action(idx)
            new_state = self._civilian_state()
            reward = self.compute_reward("civilian", state, new_state)
            self.civilian_ai.train(state, idx, reward, new_state)

    def _random_civilian_actions(self, nations: Dict[int, "Nation"]) -> None:
        """Fallback actions using AI when available, otherwise random."""
        if random.random() < 0.05:
            self.build_city()
        if self.cities and self.at_war:
            vulnerable = []
            for c in self.cities:
                risk = (50 - self.stability) / 100 if self.stability < 50 else 0.0
                for eid in self.at_war:
                    enemy = nations.get(eid)
                    if not enemy or enemy.planet != self.planet:
                        continue
                    dists_sq = [
                        distance_sq(c.coords, (d.x, d.y))
                        for d in enemy.divisions
                        if d.planet == self.planet
                    ]
                    if enemy.cities:
                        dists_sq += [distance_sq(c.coords, ec.coords) for ec in enemy.cities]
                    if dists_sq and min(dists_sq) < 75 * 75:
                        risk += 0.3
                if risk > 0 and random.random() < risk:
                    vulnerable.append(c)
            if vulnerable:
                victim = min(vulnerable, key=lambda ct: ct.population)
                victim.owner = None
                if victim in self.cities:
                    self.cities.remove(victim)

        # If an AI policy is available use it instead of random actions.
        if self.civilian_ai is not None:
            state = self._civilian_state()
            idx = self.civilian_ai.choose_action(state)
            self._execute_civilian_action(idx)
            new_state = self._civilian_state()
            reward = self.compute_reward("civilian", state, new_state)
            self.civilian_ai.train(state, idx, reward, new_state)
            return

        if random.random() < 0.03:
            self.build_base()
        if random.random() < 0.02:
            self.build_mine()
        if random.random() < 0.02:
            self.build_port()
        if random.random() < 0.02:
            self.build_factory()
        if random.random() < 0.01:
            self.build_hospital()
        if random.random() < 0.01:
            self.build_shipyard()
        if random.random() < 0.01:
            self.build_school()
        if random.random() < 0.01:
            self.build_power_plant()
        if random.random() < 0.01:
            self.build_spaceport()
        if random.random() < 0.01:
            self.build_nuke_facility()
        if random.random() < 0.01:
            self.build_orbital_defense()
        if random.random() < 0.03:
            build_division(self)
        if self.spaceports and random.random() < 0.01:
            self.colonize_planet()
        if random.random() < 0.01:
            self.build_lab()
        if random.random() < 0.01:
            self.start_project()

    def update_border_pressure(
        self,
        nations: Dict[int, "Nation"],
        bloc_map: Dict[int, "AllianceBloc"],
    ) -> None:
        """Adjust border pressure using nation or bloc metrics."""
        _diplomacy_update_border_pressure(self, nations, bloc_map)


    def finalize_turn(self) -> None:
        """Clamp key stats to sensible ranges."""
        self.population = max(0, int(self.population))
        self.economy_linear = max(0.0, self.economy_linear)
        self.stability = max(0.0, min(self.stability, 100.0))
        self.technology.science = max(1.0, self.technology.science)
        self.technology.military = max(1.0, self.technology.military)
        self.technology.industry = max(1.0, self.technology.industry)
        self.divisions = [d for d in self.divisions if d.soldiers > 0]
        self.nuclear_stockpile = max(0, self.nuclear_stockpile)
        for k in list(self.border_pressure):
            self.border_pressure[k] = max(0.0, min(self.border_pressure[k], 5.0))
        self.update_goals()

    def update_goals(self) -> None:
        """Refresh goal status and spawn new objectives."""
        self.goals.update(self)
        active = len(self.goals.active())
        if active < 2:
            self.goals.spawn_goals(self, 2 - active)
