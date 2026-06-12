"""Global runtime toggles and verbosity-aware printing.

These flags are mutated at runtime (CLI options, ``run_simulation``
arguments), so modules that need the live value must read them through the
module object (``from worldsim.core import flags`` then ``flags.VERBOSE``)
rather than importing the names directly.
"""

from __future__ import annotations

# Global toggle controlling verbosity of print statements
VERBOSE: bool = True

# Global toggle for approximate calculations
APPROXIMATE: bool = False

# Global toggle for profiling output
DEBUG_PROFILE: bool = False

# When set to a nation name string, wprint() only emits output that is
# attributed to that nation (all other nations are silenced).  Set to None
# to show output for every nation.
WATCHED_NATION: str | None = None


def vprint(*args, **kwargs) -> None:
    """Print only when ``VERBOSE`` is ``True``."""
    if VERBOSE:
        print(*args, **kwargs)


def wprint(nation_name: str, *args, **kwargs) -> None:
    """Print only when ``VERBOSE`` and the nation is being watched.

    When :data:`WATCHED_NATION` is ``None`` all nations are shown.  When it
    is set to a name string only messages attributed to that nation appear.
    This avoids building and printing strings for every nation in the
    simulation when the user cares about just one.
    """
    if VERBOSE and (WATCHED_NATION is None or WATCHED_NATION == nation_name):
        print(*args, **kwargs)


def _compact_pop(n: int) -> str:
    """Format a population count as a short human-readable string.

    Examples: ``1234567`` → ``"1.2M"``, ``45000`` → ``"45k"``, ``999`` → ``"999"``.
    """
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n // 1_000}k"
    return str(n)
