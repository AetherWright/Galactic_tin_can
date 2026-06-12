"""Wall-clock guards for simulation turns."""

from __future__ import annotations

import signal
from contextlib import contextmanager


@contextmanager
def time_limit(seconds: float | None):
    """Raise :class:`TimeoutError` if the block exceeds ``seconds``.

    On platforms without ``SIGALRM`` (e.g. Windows) or when ``seconds`` is
    ``None``/non-positive, this behaves as a no-op.
    """

    if seconds is None or seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def handler(signum, frame):  # pragma: no cover - platform specific
        raise TimeoutError("operation timed out")

    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
