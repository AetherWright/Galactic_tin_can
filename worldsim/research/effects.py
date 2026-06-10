"""Technology unlock effects.

Effects are first-class callables stored on
:class:`~worldsim.research.technology.TechnologyNode` and invoked once when a
node is unlocked.  They publish exclusively through ``nation.tech_bonuses``
(the unchanged output interface).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..nations.nation import Nation


def _mult_bonus(n: "Nation", key: str, factor: float, cap: float = 5.0) -> None:
    n.tech_bonuses[key] = min(n.tech_bonuses.get(key, 1.0) * factor, cap)


def _add_bonus(n: "Nation", key: str, amount: float) -> None:
    n.tech_bonuses[key] = n.tech_bonuses.get(key, 0.0) + amount


def _city_output(n: "Nation") -> None:
    n.tech_bonuses["city_output"] = n.tech_bonuses.get("city_output", 1.0) * 1.1


def _admin(n: "Nation") -> None:
    n.tech_bonuses["economy_mult"] = n.tech_bonuses.get("economy_mult", 1.0) * 1.1


def _math_admin(n: "Nation") -> None:
    n.tech_bonuses["economy_mult"] = n.tech_bonuses.get("economy_mult", 1.0) * 1.05


def _factory(n: "Nation") -> None:
    n.tech_bonuses["factory_output"] = n.tech_bonuses.get("factory_output", 1.0) * 1.1


def _factory_major(n: "Nation") -> None:
    n.tech_bonuses["factory_output"] = n.tech_bonuses.get("factory_output", 1.0) * 1.2


def _mine_output(n: "Nation") -> None:
    n.tech_bonuses["mine_output"] = n.tech_bonuses.get("mine_output", 1.0) * 1.1


def _atomic(n: "Nation") -> None:
    n.tech_bonuses["nuclear_prod"] = 1.0


def _nuclear(n: "Nation") -> None:
    n.tech_bonuses["nuclear_power"] = 1.0


# -- Biology: Ecology track -------------------------------------------------
def _botany(n: "Nation") -> None:
    _mult_bonus(n, "city_output", 1.05)
    _add_bonus(n, "plague_resist", 0.05)


def _ecosystem_mapping(n: "Nation") -> None:
    # Signals terraforming readiness and improves colony viability.
    n.tech_bonuses["terraform_readiness"] = 1.0
    _mult_bonus(n, "colony_viability", 1.1)


def _synthetic_ecology(n: "Nation") -> None:
    _mult_bonus(n, "habitability", 1.15)


def _terraforming(n: "Nation") -> None:
    # Flag consumed by colonization to unlock new colony types.
    n.tech_bonuses["terraforming"] = 1.0


# -- Biology: Medicine track ------------------------------------------------
def _herbalism(n: "Nation") -> None:
    _add_bonus(n, "plague_resist", 0.1)


def _medicine(n: "Nation") -> None:
    _add_bonus(n, "plague_resist", 0.2)
    _mult_bonus(n, "hospital_effectiveness", 1.1)


def _pharmacology(n: "Nation") -> None:
    _add_bonus(n, "plague_resist", 0.3)
    _mult_bonus(n, "pop_growth", 1.1)


def _advanced_medicine(n: "Nation") -> None:
    _add_bonus(n, "plague_resist", 0.4)


def _genetic_medicine(n: "Nation") -> None:
    _add_bonus(n, "plague_resist", 0.5)
    _mult_bonus(n, "resilience_bonus", 1.1)


# -- Biology: Genetics track ------------------------------------------------
def _cell_biology(n: "Nation") -> None:
    _mult_bonus(n, "research_mult", 1.1)
    _mult_bonus(n, "lab_effectiveness", 1.1)


def _genetic_engineering(n: "Nation") -> None:
    _mult_bonus(n, "pop_cap", 1.1)
    _mult_bonus(n, "mutation_rate", 1.1)


def _synthetic_biology(n: "Nation") -> None:
    _mult_bonus(n, "food_output", 1.15)


def _directed_evolution(n: "Nation") -> None:
    _mult_bonus(n, "adaptive_colonization", 1.1)

