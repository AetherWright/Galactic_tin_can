"""Bridge to the ``rust_events`` crate's Q-table engine.

The crate ships a small binary (``rust_event_engine``) that keeps its own
YAML Q-table and speaks JSON over stdin:

``choose``  pick the best option index for ``(event, state)``
``update``  TD(0)-update the table with an observed reward

The bridge builds the binary on first use when ``cargo`` is available and
disables itself permanently for the run after any failure, so the Python
Q-learner (which keeps learning in parallel) takes over seamlessly.

Because the binary loads and rewrites its table file on every call, the
bridge is opt-in: enable with ``--rust-events`` on the CLI,
``use_rust_events=True`` on :class:`~worldsim.events.engine.EventDecisionEngine`,
or ``WORLDSIM_USE_RUST_EVENTS=1``.
"""

from __future__ import annotations

import json
import multiprocessing as _mp
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

from ..core.native import NATIVE_DIR

_CRATE_DIR = NATIVE_DIR / "rust_events"
_BINARY = _CRATE_DIR / "target" / "release" / "rust_event_engine"

#: Wall-clock guard per subprocess call.
_CALL_TIMEOUT_S: float = 5.0


def _build_binary() -> None:
    if _BINARY.exists() or _mp.current_process().name != "MainProcess":
        return
    if shutil.which("cargo") is None or not (_CRATE_DIR / "Cargo.toml").exists():
        return
    try:
        subprocess.run(["cargo", "build", "--release"], cwd=_CRATE_DIR, check=False)
    except Exception:
        return


class RustEventBridge:
    """Stateful wrapper around the ``rust_event_engine`` binary."""

    def __init__(self, qtable_path: str | Path) -> None:
        base = Path(qtable_path)
        if base.suffix.lower() == ".yml":
            base = base.with_suffix("")
        # The crate keeps its own table format — separate file from the
        # per-event YAML tables written by the Python EventQLearner.
        self.table_path = Path(f"{base}_rust.yml")
        self._dead = False

    @classmethod
    def create(cls, qtable_path: str | Path) -> Optional["RustEventBridge"]:
        """Return a working bridge, building the binary if needed, else ``None``."""
        _build_binary()
        if not _BINARY.exists():
            return None
        bridge = cls(qtable_path)
        return bridge if bridge._ping() else None

    # ------------------------------------------------------------------

    def _call(self, mode: str, payload: dict) -> Optional[str]:
        if self._dead:
            return None
        try:
            result = subprocess.run(
                [str(_BINARY), mode, str(self.table_path)],
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                timeout=_CALL_TIMEOUT_S,
            )
        except Exception:
            self._dead = True
            return None
        if result.returncode != 0:
            self._dead = True
            return None
        return result.stdout.strip()

    def _ping(self) -> bool:
        return self._call("ping", {}) == "ok"

    @property
    def alive(self) -> bool:
        return not self._dead

    # ------------------------------------------------------------------

    def choose(self, event: str, n_options: int, state: List[float]) -> Optional[int]:
        """Return the crate's option index for ``(event, state)``, or ``None``."""
        if n_options <= 0:
            return None
        out = self._call(
            "choose",
            {"event": event, "options": list(range(n_options)), "state": state},
        )
        if out is None:
            return None
        try:
            idx = int(out)
        except ValueError:
            self._dead = True
            return None
        if 0 <= idx < n_options:
            return idx
        return None

    def update(
        self,
        event: str,
        state: List[float],
        action: int,
        reward: float,
        new_state: List[float],
    ) -> None:
        """Feed an observed transition back into the crate's Q-table."""
        self._call(
            "update",
            {
                "event": event,
                "action": int(action),
                "reward": float(reward),
                "state": state,
                "new_state": new_state,
            },
        )
