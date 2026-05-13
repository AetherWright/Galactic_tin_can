"""Text-based console manager for WorldSim.

The :class:`ConsoleUI` class creates a top panel for overall world statistics
and additional panels for each nation.  Each nation panel displays a rolling
log of recent events.  When :mod:`curses` is available it is used to create
separate windows; otherwise the class falls back to plain text output which
works on basic Windows consoles.
"""

from __future__ import annotations

from typing import Dict, List
import sys

# ``curses`` is not part of the default Windows Python distribution.  Import it
# conditionally so the simulation can still run in environments where it is
# missing.
try:  # pragma: no cover - exercised only on systems without curses
    import curses  # type: ignore[import]
    _HAS_CURSES = True
except Exception:  # pragma: no cover - import failure path
    curses = None  # type: ignore[assignment]
    _HAS_CURSES = False


class ConsoleUI:
    """Manage terminal output for world and nation information.

    If :mod:`curses` is available the UI splits the screen into multiple
    windows.  Otherwise it prints simple text logs, providing basic compatibility
    with Windows consoles.

    Parameters
    ----------
    nation_names:
        Names of nations to allocate individual log windows for.
    """

    def __init__(self, nation_names: List[str]) -> None:
        self.nation_names = nation_names
        self._logs: Dict[str, List[str]] = {}
        # Use curses only if it is available and the process is attached to a TTY.
        self._use_curses = _HAS_CURSES and sys.stdin.isatty() and sys.stdout.isatty()
        self.nation_wins: Dict[str, "curses._CursesWindow"] = {}
        if self._use_curses:
            try:  # pragma: no cover - requires terminal
                self.screen = curses.initscr()
                curses.noecho()
                curses.cbreak()
                self.screen.keypad(True)
                height, width = self.screen.getmaxyx()
                self.world_win = curses.newwin(3, width, 0, 0)
                remaining_height = height - 3
                if nation_names:
                    each_height = max(1, remaining_height // len(nation_names))
                    for idx, name in enumerate(nation_names):
                        win = curses.newwin(each_height, width, 3 + idx * each_height, 0)
                        win.border()
                        win.addstr(0, 2, f" {name} ")
                        self.nation_wins[name] = win
                self.world_win.border()
                self.world_win.addstr(0, 2, " World ")
                self.refresh()
            except Exception:  # pragma: no cover - init failure path
                # Fall back to plain output if curses cannot initialize (e.g.
                # unsupported terminals)
                self._use_curses = False
                print(
                    "ConsoleUI: curses initialization failed; using plain console output.")
        else:
            print("ConsoleUI: curses not available or no TTY; using plain console output.")

    # REVIEW: additional helper methods may be added for richer UI later

    def refresh(self) -> None:
        """Refresh all windows."""
        if self._use_curses:  # pragma: no branch - simple guard
            self.world_win.noutrefresh()
            for win in self.nation_wins.values():
                win.noutrefresh()
            curses.doupdate()

    def update_world(self, lines: List[str]) -> None:
        """Display world-level statistics."""
        if self._use_curses:  # pragma: no branch - simple guard
            self.world_win.erase()
            self.world_win.border()
            self.world_win.addstr(0, 2, " World ")
            max_y, max_x = self.world_win.getmaxyx()
            for i, line in enumerate(lines[: max_y - 2]):
                self.world_win.addstr(1 + i, 1, line[: max_x - 2])
            self.refresh()
        else:
            print(" | ".join(lines))

    def append_nation_log(self, nation: str, line: str) -> None:
        """Append ``line`` to the log window for ``nation``."""
        if self._use_curses:  # pragma: no branch - simple guard
            win = self.nation_wins.get(nation)
            if not win:
                return
            log = self._logs.setdefault(nation, [])
            log.append(line)
            max_y, max_x = win.getmaxyx()
            if len(log) > max_y - 2:
                del log[0]
            win.erase()
            win.border()
            win.addstr(0, 2, f" {nation} ")
            for i, ln in enumerate(log):
                win.addstr(1 + i, 1, ln[: max_x - 2])
            self.refresh()
        else:
            print(f"[{nation}] {line}")

    def close(self) -> None:
        """Restore terminal state."""
        if self._use_curses:  # pragma: no branch - simple guard
            curses.nocbreak()
            self.screen.keypad(False)
            curses.echo()
            curses.endwin()
