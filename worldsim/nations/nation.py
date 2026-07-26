"""The Nation: per-nation state and the per-fifth ``process_turn`` pipeline.

The class itself stays thin — economy, war, diplomacy, research, projects,
construction and the civilian AI all live in their own arms; the wrapper
methods here delegate so call sites can stay ``nation.do_thing()``.
"""

from dataclasses import dataclass, field
from typing import Dict, Set, List, Tuple, Optional, Any, ClassVar
from pathlib import Path
import random
import math

from ..ai import (
    DomesticPolicyAI, ProjectAI, DiplomacyAI, WarAI,
    CivilianOverseerAI, DepartmentPolicyAI,
)
from ..diplomacy import (
    Message,
    can_communicate,
    send_message           as _diplomacy_send_message,
    process_messages       as _diplomacy_process_messages,
    supply_nation          as _diplomacy_supply_nation,
    update_border_pressure as _diplomacy_update_border_pressure,
    compute_trade_bonus    as _diplomacy_trade_bonus,
    collect_tributes       as _diplomacy_collect_tributes,
    get_all_allies         as _diplomacy_get_all_allies,
)
from .economy import Economy
from ..society.culture import Culture
from ..society.ideas import Idea
from ..society.leader import Leader, LeaderModel
from ..society.goals import GoalManager, Goal

from ..core import (
    polygon_centroid,
    polygon_area,
    distance_sq,
    _np,
)
from ..config import load_list

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
    Farm,
    Port,
    Factory,
    Hospital,
    Shipyard,
    Spaceport,
    NuclearFacility,
    OrbitalDefense,
)
from ..research import Technology, TechnologyTree, setup_default_tech_tree
from ..military.divisions import Division, DivisionTemplate, build_division
from ..military.nuclear import (
    produce_nuclear_weapons  as _war_produce_nuclear_weapons,
    launch_nuclear_strike    as _war_launch_nuclear_strike,
    launch_first_strike      as _war_launch_first_strike,
    consider_first_strike    as _war_consider_first_strike,
)
from ..military.fleets import Fleet, build_fleet_ship, process_fleet_movement, order_fleet_move
from ..military.doctrine import DoctrineAI
from ..military.command import issue_doctrine
from ..galaxy.star import Star, STARS
from .government import Government
from .projects import (
    NationalProject,
    ProjectSpec,
    PROJECT_CATALOG,
    PROJECT_NAMES,
    available_projects  as _proj_available_projects,
    start_project       as _proj_start_project,
    progress_projects   as _proj_progress_projects,
)
from .construction import (
    RESOURCE_COSTS,
    collect_resources   as _infra_collect_resources,
    build_city          as _infra_build_city,
    build_base          as _infra_build_base,
    build_mine          as _infra_build_mine,
    build_farm          as _infra_build_farm,
    build_port          as _infra_build_port,
    build_factory       as _infra_build_factory,
    build_hospital      as _infra_build_hospital,
    build_shipyard      as _infra_build_shipyard,
    build_school        as _infra_build_school,
    build_power_plant   as _infra_build_power_plant,
    build_lab           as _infra_build_lab,
    build_nuke_facility as _infra_build_nuke_facility,
    build_orbital_defense as _infra_build_orbital_defense,
    build_spaceport     as _infra_build_spaceport,
    upgrade_assets      as _infra_upgrade_assets,
    colonize_planet     as _infra_colonize_planet,
)
from .civilian import (
    _civilian_state          as _cc_civilian_state,
    _execute_civilian_action as _cc_execute_civilian_action,
    _valid_action_mask       as _cc_valid_action_mask,
    _apply_civilian_ai       as _cc_apply_civilian_ai,
    _random_civilian_actions as _cc_random_civilian_actions,
    process_action_queue     as _cc_process_action_queue,
    CivilianController,
    DEPARTMENTS,
    N_CIVILIAN_DEPARTMENTS,
)


def _iter_rnn_controllers(ctrl: Any):
    """Yield every underlying unified-trunk role view for *ctrl*.

    A :class:`CivilianController` is hierarchical: it owns an overseer plus
    one model per department.  Expanding it here lets meta-evolution reach
    each per-nation head LoRA, rather than skipping the wrapper entirely.
    Any non-torch (NelderMead / legacy
    :class:`~worldsim.events.qlearner.EventQLearner`) or ``None`` controller
    yields nothing.
    """
    try:
        from ..ai.rnn import UnifiedRoleView
    except Exception:
        return
    if ctrl is None:
        return
    if isinstance(ctrl, CivilianController):
        candidates = (ctrl.overseer, *ctrl.dept_models.values())
    else:
        candidates = (ctrl,)
    for c in candidates:
        if isinstance(c, UnifiedRoleView):
            yield c



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
    farms: List[Farm] = field(default_factory=list)
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
    # Shared categorical (C51) event Q-learning agent, LoRA-specialised per
    # nation — ``None`` when torch is unavailable, in which case
    # ``EventDecisionEngine`` falls back to its own tabular ``EventQLearner``.
    events_ai: Optional[Any] = field(default=None)
    # This nation's shared unified-trunk LoRA bank (core LoRA + one head LoRA
    # per role it uses) — ``None`` when torch is unavailable, in which case
    # every *_ai attribute above is a legacy (non-torch) controller instead.
    _unified_bank: Optional[Any] = field(default=None, repr=False)
    # This nation's batched ground-division LoRA bank (see
    # worldsim.ai.rnn.divisions.DivisionLoRABank) — None when torch is
    # unavailable, in which case each Division falls back to its own tiny
    # legacy posture/movement perceptron pair.
    _division_bank: Optional[Any] = field(default=None, repr=False)
    action_queue: List[int] = field(default_factory=list)
    inbox: List[Message] = field(default_factory=list)
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
        military_path  = self._ai_table_path("military")
        civilian_path  = self._ai_table_path("civilian")
        project_path   = self._ai_table_path("projects")
        diplomacy_path = self._ai_table_path("diplomacy")
        doctrine_path  = self._ai_table_path("doctrine")
        events_path    = self._ai_table_path("events")

        # Prefer the PyTorch RNN backend when available; fall back to the
        # hand-rolled Nelder-Mead controllers otherwise.
        try:
            from ..ai.rnn import (
                rnn_available,
                TorchWarAI,
                TorchCivilianOverseer,
                TorchDepartmentAI,
                TorchProjectAI,
                TorchDiplomacyAI,
                TorchDoctrineAI,
                TorchEventAI,
                UnifiedLoRABank,
                DivisionLoRABank,
            )
            from ..military.divisions import DIVISION_N_INPUTS, DIVISION_N_ACTIONS
            _use_torch = rnn_available()
        except Exception:  # ImportError or anything else
            _use_torch = False

        if _use_torch:
            bank = UnifiedLoRABank(core_state_fn=self.unified_core_state)
            self._unified_bank = bank
            self._division_bank = DivisionLoRABank(DIVISION_N_INPUTS, DIVISION_N_ACTIONS)
            self.military_ai  = TorchWarAI(
                allies_dim=ally_dim, table_path=military_path, bank=bank,
            )
            self.civilian_ai  = CivilianController(
                overseer=TorchCivilianOverseer(
                    n_depts=N_CIVILIAN_DEPARTMENTS, n_inputs=22, bank=bank,
                ),
                dept_models={
                    dept.slug: TorchDepartmentAI(
                        dept.slug, dept.n_actions, n_inputs=22, bank=bank,
                    )
                    for dept in DEPARTMENTS
                },
                base_table_path=civilian_path,
            )
            self.project_ai   = TorchProjectAI(
                len(PROJECT_CATALOG), n_inputs=22, table_path=project_path, bank=bank,
            )
            self.diplomacy_ai = TorchDiplomacyAI(
                3, n_inputs=5, table_path=diplomacy_path, bank=bank,
            )
            self.doctrine_ai  = TorchDoctrineAI(table_path=doctrine_path, bank=bank)
            self.events_ai    = TorchEventAI(table_path=events_path, bank=bank)
        else:
            self.military_ai  = WarAI(allies_dim=ally_dim, table_path=military_path)
            self.civilian_ai  = CivilianController(
                overseer=CivilianOverseerAI(
                    n_depts=N_CIVILIAN_DEPARTMENTS, n_inputs=22,
                ),
                dept_models={
                    dept.slug: DepartmentPolicyAI(
                        dept.slug, dept.n_actions, n_inputs=22,
                    )
                    for dept in DEPARTMENTS
                },
                base_table_path=civilian_path,
            )
            self.project_ai   = ProjectAI(
                len(PROJECT_CATALOG), n_inputs=6, table_path=project_path
            )
            self.diplomacy_ai = DiplomacyAI(3, n_inputs=5, table_path=diplomacy_path)
            # DoctrineAI runs AFTER civilian and diplomacy AIs each fifth
            self.doctrine_ai  = DoctrineAI(table_path=doctrine_path)

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

    # ------------------------------------------------------------------
    # Upkeep & civilian build budget
    # ------------------------------------------------------------------
    # Per-turn maintenance drawn against gross output, by asset class.  These
    # are deliberately small: their job is to make net income (and therefore
    # the civilian build budget) shrink as an empire grows, so expansion is
    # self-limiting rather than to bankrupt nations.
    _UPKEEP_BUILDING: ClassVar[float] = 0.5
    _UPKEEP_DIVISION: ClassVar[float] = 1.0
    _UPKEEP_SHIP:     ClassVar[float] = 1.5

    def _building_count(self) -> int:
        return (
            len(self.cities) + len(self.bases) + len(self.mines) + len(self.farms)
            + len(self.ports) + len(self.factories) + len(self.hospitals)
            + len(self.shipyards) + len(self.spaceports) + len(self.schools)
            + len(self.power_plants) + len(self.labs) + len(self.nuke_plants)
            + len(self.orbital_defenses)
        )

    def compute_upkeep(self) -> float:
        """Total per-turn upkeep for all of the nation's standing assets."""
        return (
            self._UPKEEP_BUILDING * self._building_count()
            + self._UPKEEP_DIVISION * len(self.divisions)
            + self._UPKEEP_SHIP * self._fleet_count()
        )

    def civilian_action_budget(self) -> float:
        """Soft per-turn budget the civilian AI may spend on construction.

        Net income plus a slice of the treasury (see
        :meth:`Economy.civilian_budget`).  ``settle_turn`` must have run for
        the current turn first (it does, in :meth:`process_turn`).
        """
        return self.econ.civilian_budget()

    def draw_civilian_budget(self, amount: float) -> None:
        """Deduct committed civilian construction spend from the treasury."""
        self.econ.draw_reserve(amount)

    @property
    def reserves(self) -> float:
        return self.econ.reserves

    @property
    def last_net_income(self) -> float:
        return self.econ.last_net_income

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
        self.division_templates["Infantry"] = DivisionTemplate("Infantry", 1000)
        ally_dim = max(1, len(self.all_ids)) if self.all_ids else 1
        # Build the AI controllers (and this nation's shared unified LoRA
        # bank, when torch is available) before the tech tree, so the
        # research subsystems' attach_ai() can bind to the same bank as
        # every other role.
        self._init_ai_controllers(ally_dim)
        setup_default_tech_tree(self)
        self.doctrine = self.military_ai.create_doctrine()
        if self.doctrine not in self.available_doctrines:
            self.available_doctrines.append(self.doctrine)
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

    # Fixed weighting applied to a :meth:`reward_snapshot` delta by
    # :meth:`compute_reward` — hand-designed once, shared by every
    # directorate (research/projects/civilian/diplomacy/events), rather than
    # a separately GA-evolved vector per directorate.  Relative emphasis
    # mirrors ``engine.scoring.century_score``: stability weighted heaviest,
    # star/fleet counts next (they change rarely, so a unit change should
    # matter a lot), economy/military/technology moderate, population scaled
    # down since its deltas run into the thousands.
    _REWARD_WEIGHTS: ClassVar[Tuple[float, ...]] = (
        0.05,    # economy (log-scaled)
        1.0,     # stability
        0.1,     # military
        0.3,     # technology.overall
        0.1,     # infrastructure
        0.0005,  # population
        5.0,     # star_count
        2.0,     # fleet count
    )

    def reward_snapshot(self) -> List[float]:
        """Fixed-order outcome vector consumed by :meth:`compute_reward`.

        Independent of any directorate's own observation-vector shape —
        every call site captures this before and after acting so
        :meth:`compute_reward` can apply one designed weighting identically
        for every directorate, rather than each directorate reading its own
        differently-shaped state vector.

        Uses the log-scaled ``economy`` (not ``economy_linear``, which is
        unbounded/exponential) — matching every existing observation vector
        (``_civilian_state``, ``EventDecisionEngine._state_snapshot``,
        ``century_score``) so a single wealthy nation can't produce a reward
        spike large enough to destabilise training.
        """
        return [
            self.economy,
            self.stability,
            self.military,
            self.technology.overall,
            self.infrastructure,
            float(self.population),
            float(self.star_count),
            self._fleet_count(),
        ]

    def unified_core_state(self) -> List[float]:
        """Canonical 16-feature input every unified-trunk role sees.

        The same 8 outcome fields as :meth:`reward_snapshot`, NN-scaled
        instead of reward-scaled, plus the nation's 8
        :class:`~worldsim.society.culture.Culture` traits (already 0..1) in
        ``Culture.__dataclass_fields__`` order — this is what lets every
        role's decision (war, civilian, diplomacy, research, ...) condition
        on the nation's culture: see
        :mod:`worldsim.ai.rnn.unified` for how it's consumed.
        """
        return [
            self.economy / 100.0,
            self.stability / 100.0,
            self.military / 100.0,
            self.technology.overall / 100.0,
            self.infrastructure / 100.0,
            float(self.population) / 1_000_000.0,
            float(self.star_count) / 10.0,
            self._fleet_count() / 10.0,
            *self.culture.asdict().values(),
        ]

    def compute_reward(self, before: List[float], after: List[float]) -> float:
        """Return the reward for a *before* → *after* :meth:`reward_snapshot` pair.

        One fixed, hand-designed weighting (:attr:`_REWARD_WEIGHTS`) shared
        by every directorate, plus a small goal-progress bonus so the RL
        signal explicitly rewards progress toward the nation's current
        objectives.
        """
        diff = [a - b for a, b in zip(after, before)]
        base_reward = sum(w * d for w, d in zip(self._REWARD_WEIGHTS, diff))

        # progress() is O(1) per goal (pure arithmetic on nation attributes).
        goal_bonus = sum(
            g.progress(self) * g.priority for g in self.goals.active()
        ) * 0.01

        return base_reward + goal_bonus

    def step_meta(self, year: int) -> None:
        self.current_year = year
        self.leader_model.step()

    def evolve_meta(self) -> None:
        self.leader_model.evolve()
        self.leader = self.leader_model.generate(self)
        self.last_collapse = self.current_year
        # Mutate LoRA + reset optimizers for every role view (and the shared
        # core LoRA) so the behaviour space is explored on collapse/rebirth.
        # Silent failure when torch is absent or controllers are the legacy
        # NelderMead type.
        try:
            if self._unified_bank is not None:
                self._unified_bank.mutate_core(sigma=0.02)
                self._unified_bank.reset_core_optimizer()
            if self._division_bank is not None:
                self._division_bank.mutate_lora(sigma=0.02)
                self._division_bank.reset_optimizer()
            _CTRL_ATTRS = (
                "civilian_ai", "military_ai", "project_ai",
                "diplomacy_ai", "doctrine_ai", "events_ai",
            )
            for attr in _CTRL_ATTRS:
                ctrl = getattr(self, attr, None)
                # The civilian AI is a hierarchical CivilianController wrapping
                # an overseer plus one model per department; expand it so each
                # underlying role view is evolved.
                for sub in _iter_rnn_controllers(ctrl):
                    sub.mutate_lora(sigma=0.02)
                    sub.reset_lora_optimizer()
            # Research subsystems (physics/engineering/biology) live under
            # tech_tree (a ResearchDirector), not a direct Nation attribute.
            for sub in getattr(self.tech_tree, "subsystems", ()):
                for view in _iter_rnn_controllers(getattr(sub, "ai", None)):
                    view.mutate_lora(sigma=0.02)
                    view.reset_lora_optimizer()
        except Exception:
            pass

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
        return _diplomacy_get_all_allies(self, nations)

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
        _infra_collect_resources(self)

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
        plague_level    = planet.plague_level    if planet else 0.0
        radiation_level = planet.radiation_level if planet else 0.0
        cbonus = self.tech_bonuses.get("city_output", 1.0)

        # Infrastructure investment health: how much the nation can afford to
        # invest in city upkeep relative to its city count.  Normalised to 1.0
        # meaning "fully funded" (economy_linear ≥ 50 per city).
        econ_health = min(
            1.0,
            self.economy_linear / max(1.0, len(self.cities) * 50.0)
        ) if self.cities else 0.3

        pop, econ = process_city_batch(
            self.cities, plague_level, radiation_level, plague_res, cbonus,
            self.tech_bonuses.get("pop_growth", 1.0),   # biology: Pharmacology
            self.tech_bonuses.get("pop_cap", 1.0),      # biology: Genetic Engineering
            stability=self.stability,
            hospital_count=len(self.hospitals),
            econ_health=econ_health,
        )
        total_pop += pop
        total_econ += econ

        # Rural migration: pull rural residents into cities with spare capacity.
        from ..planets.demographics.city import process_rural_migration
        process_rural_migration(self.cities)

        if self.colonies:
            cpop = process_colony_batch(
                self.colonies, plague_level, plague_res,
                stability=self.stability,
                radiation_level=radiation_level,
            )
            total_pop += cpop

        # Rural population dynamics for owned counties on the home planet.
        if planet is not None:
            food_ratio = min(
                1.0,
                planet.resources.get('food', 20.0) / 20.0,
            )
            for county in planet.counties.values():
                if county.owner == self.id:
                    county.process_rural_turn(
                        food_ratio=food_ratio,
                        stability=self.stability,
                    )
                    total_pop += county.rural_population
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
        # biology: Cell Biology boosts lab yield (lab_effectiveness)
        lab_eff = self.tech_bonuses.get("lab_effectiveness", 1.0)
        if _np is not None and self.labs:
            arr = _np.array([l.output for l in self.labs], dtype=float)
            research_bonus += float(arr.sum()) * lab_eff
        else:
            research_bonus += sum(l.output for l in self.labs) * lab_eff
        # biology: Cell Biology general research multiplier (research_mult)
        research_bonus *= self.tech_bonuses.get("research_mult", 1.0)

        # biology: Medicine boosts hospital plague suppression (hospital_effectiveness)
        hosp_eff = self.tech_bonuses.get("hospital_effectiveness", 1.0)
        if _np is not None and self.hospitals:
            levels = _np.array([hos.level for hos in self.hospitals], dtype=float)
            for hos, level in zip(self.hospitals, levels.tolist()):
                planet = PLANETS.get(hos.planet)
                if planet:
                    planet.plague_level = max(0.0, planet.plague_level - level * 0.005 * hosp_eff)
        else:
            for hos in self.hospitals:
                planet = PLANETS.get(hos.planet)
                if planet:
                    planet.plague_level = max(0.0, planet.plague_level - hos.level * 0.005 * hosp_eff)

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

        # Bank net income (gross output minus upkeep) into the treasury so the
        # civilian AI has a soft per-turn construction budget to spend against.
        self.econ.settle_turn(self.economy_linear, self.compute_upkeep())

        self.tech_tree.research(
            self.economy * 0.05 + research_bonus,
            self,
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
            from ..society.culture import ARCHETYPE_IDEALS
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
        _infra_build_city(self)

    def build_base(self) -> None:
        _infra_build_base(self)

    def build_mine(self) -> None:
        _infra_build_mine(self)

    def build_farm(self) -> None:
        _infra_build_farm(self)

    def build_port(self) -> None:
        _infra_build_port(self)

    def build_factory(self) -> None:
        _infra_build_factory(self)

    def build_hospital(self) -> None:
        _infra_build_hospital(self)

    def build_shipyard(self) -> None:
        _infra_build_shipyard(self)

    def build_school(self) -> None:
        _infra_build_school(self)

    def build_power_plant(self) -> None:
        _infra_build_power_plant(self)

    def build_lab(self) -> None:
        _infra_build_lab(self)

    def build_nuke_facility(self) -> None:
        _infra_build_nuke_facility(self)

    def build_orbital_defense(self) -> None:
        _infra_build_orbital_defense(self)

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
        _infra_build_spaceport(self)

    def colonize_planet(self) -> None:
        """Attempt to found a city on another planet if one is free."""
        _infra_colonize_planet(self)

    def upgrade_assets(self) -> None:
        """Invest economy into upgrading owned infrastructure."""
        _infra_upgrade_assets(self)

    def produce_nuclear_weapons(self) -> None:
        """Convert uranium, metal and economy into nuclear stockpile."""
        _war_produce_nuclear_weapons(self)

    def launch_nuclear_strike(self, enemy: "Nation") -> None:
        """Inflict heavy losses on ``enemy`` using one warhead."""
        _war_launch_nuclear_strike(self, enemy)

    def launch_first_strike(self, enemies: List["Nation"]) -> None:
        """Launch all available warheads across ``enemies`` in round-robin fashion."""
        _war_launch_first_strike(self, enemies)

    def consider_first_strike(self, nations: Dict[int, "Nation"]) -> None:
        """Allow the military AI to initiate a nuclear first strike."""
        _war_consider_first_strike(self, nations)

    def available_projects(self) -> List[str]:
        """Return project names that can currently be started."""
        return _proj_available_projects(self)

    def start_project(self, name: Optional[str] = None) -> None:
        """Begin a new national project if resources allow."""
        _proj_start_project(self, name)

    def progress_projects(self) -> None:
        """Spend economy to advance national projects."""
        _proj_progress_projects(self)
    @property
    def star_count(self) -> int:
        return sum(1 for s in STARS.values() if s.owner == self.id)

    def _civilian_state(self) -> List[float]:
        return _cc_civilian_state(self)

    def _execute_civilian_action(self, idx: int) -> None:
        _cc_execute_civilian_action(self, idx)

    def _valid_action_mask(self) -> List[bool]:
        return _cc_valid_action_mask(self)

    def _apply_civilian_ai(self) -> None:
        _cc_apply_civilian_ai(self)

    def process_action_queue(self, limit: int = 2) -> None:
        _cc_process_action_queue(self, limit)

    def _random_civilian_actions(self, nations: Dict[int, "Nation"]) -> None:
        _cc_random_civilian_actions(self, nations)

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
