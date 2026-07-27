"""Tests for civil-war / ally-war-entry dampening.

A real run showed civil wars compounding almost unboundedly (4 starting
nations turning into 150+ within ~80 centuries) because
``_handle_internal_conflicts``'s per-nation risk was uncapped (near-certain
at low stability) and rolled once per *fifth* (5x/century), the civil-war
"parent" nation never got its own post-split cooldown, and the class-based
``SimulationLoop`` was missing the same zombie-nation cap the CLI entry
point (``run_simulation``) already had. Separately, ally war-entry always
rolled the same flat chance regardless of how much of the galaxy was
already fighting.
"""
from __future__ import annotations

import random

from worldsim.diplomacy.wargoals import (
    ALLY_WAR_ENTRY_CHANCE,
    _war_weariness_factor,
)
from worldsim.engine.politics import (
    _MAX_CIVIL_WAR_RISK_PER_FIFTH,
    _handle_internal_conflicts,
    _spawn_rebel_nation,
)
from worldsim.galaxy import init_world


def make_world(n=3, stars=1, pop=5000):
    return init_world(n, stars, starting_population=pop)


class TestCivilWarRiskCap:
    def test_risk_is_capped_even_at_zero_stability_and_military(self):
        """Force the worst-case inputs (risk formula would otherwise exceed
        1.0) and confirm the observed spawn rate over many trials tracks the
        capped ceiling, not the uncapped raw formula."""
        random.seed(0)
        trials = 400
        spawns = 0
        for _ in range(trials):
            nations = make_world(2, 1, 20000)
            n = next(iter(nations.values()))
            n.stability = 0.0
            n.military = 0.0
            n.last_collapse = -1000  # long past the 10-year grace period
            before = set(nations.keys())
            _handle_internal_conflicts(nations, year=2000)
            if set(nations.keys()) != before:
                spawns += 1
        rate = spawns / trials
        # Uncapped, this scenario rolls at ~100% every trial. The cap keeps
        # it at _MAX_CIVIL_WAR_RISK_PER_FIFTH; allow generous statistical
        # slack (the formula has other additive terms) but it must be
        # nowhere near the old near-certain behavior.
        assert rate < _MAX_CIVIL_WAR_RISK_PER_FIFTH + 0.05, (
            f"observed civil war rate {rate:.2f} exceeds the intended cap"
        )

    def test_parent_gets_its_own_cooldown_after_a_split(self):
        nations = make_world(2, 1, 20000)
        parent = next(iter(nations.values()))
        parent.stability = 10.0
        year = 500
        _spawn_rebel_nation(parent, nations, year)
        assert parent.last_collapse == year


class TestAllyWarWeariness:
    def test_no_wars_gives_full_chance(self):
        nations = make_world(5, 1, 5000)
        for n in nations.values():
            n.at_war.clear()
        assert _war_weariness_factor(nations) == 1.0

    def test_more_wars_reduces_chance(self):
        nations = make_world(6, 1, 5000)
        ids = list(nations.keys())
        for n in nations.values():
            n.at_war.clear()
        full = _war_weariness_factor(nations)
        nations[ids[0]].at_war.add(ids[1])
        nations[ids[1]].at_war.add(ids[0])
        nations[ids[2]].at_war.add(ids[3])
        nations[ids[3]].at_war.add(ids[2])
        partial = _war_weariness_factor(nations)
        assert partial < full

    def test_factor_never_drops_below_floor(self):
        nations = make_world(4, 1, 5000)
        for n in nations.values():
            n.at_war = {99999}  # everyone "at war"
        factor = _war_weariness_factor(nations)
        assert factor >= 0.15

    def test_empty_nations_is_full_chance(self):
        assert _war_weariness_factor({}) == 1.0


class TestSimulationLoopCullsZombies:
    def test_step_culls_down_to_max_nations(self):
        from worldsim.engine.loop import SimulationLoop

        nations = make_world(3, 1, 5000)
        # Fabricate a pile of zero-population, zero-city micro-states —
        # exactly what _cull_zombie_nations's first pass removes outright.
        next_id = max(nations) + 1
        for i in range(60):
            from worldsim.nations import Nation
            from worldsim.society.culture import Culture

            n = Nation(f"Micro{i}", next_id + i, Culture.random(), [])
            n.population = 0
            nations[n.id] = n

        loop = SimulationLoop(nations, log_path=None)
        loop.step()
        assert len(loop.nations) <= 53  # 3 real + at most a few stragglers this pass
