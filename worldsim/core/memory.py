"""RAM / VRAM probing and memory-budgeted batch sizing.

Every batch-shaped hot path (replay training, city demographics, star
ownership fan-out) sizes its chunks from the memory actually available
instead of fixed constants, so small machines degrade gracefully and large
ones use their headroom.

Budgets
-------
A *budget* is ``available_memory × fraction``, clamped by the optional
environment overrides:

``WORLDSIM_RAM_BUDGET_MB``   absolute cap for host-RAM budgets
``WORLDSIM_VRAM_BUDGET_MB``  absolute cap for GPU budgets
``WORLDSIM_MEM_FRACTION``    override the per-call ``fraction`` everywhere

When a probe is unavailable (no ``/proc/meminfo``, no GPU runtime) the
helpers fall back to the caller-supplied ``default`` so behaviour matches
the previous fixed-constant code exactly.
"""

from __future__ import annotations

import os
from typing import Iterator, Optional, Sequence

_MEMINFO = "/proc/meminfo"


def _env_mb(name: str) -> Optional[int]:
    raw = os.getenv(name)
    if not raw:
        return None
    try:
        return int(float(raw) * 1024 * 1024)
    except ValueError:
        return None


def _env_fraction() -> Optional[float]:
    raw = os.getenv("WORLDSIM_MEM_FRACTION")
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    if 0.0 < value <= 1.0:
        return value
    return None


def available_ram_bytes() -> Optional[int]:
    """Return ``MemAvailable`` in bytes, or ``None`` when unknowable."""
    try:
        with open(_MEMINFO, "r", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    try:  # pragma: no cover - optional dependency
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        return None


def available_vram_bytes() -> Optional[int]:
    """Return free GPU memory in bytes via torch or CuPy, or ``None``."""
    try:  # pragma: no cover - GPU-only path
        import torch
        if torch.cuda.is_available():
            free, _total = torch.cuda.mem_get_info()
            return int(free)
    except Exception:
        pass
    try:  # pragma: no cover - GPU-only path
        import cupy
        free, _total = cupy.cuda.runtime.memGetInfo()
        return int(free)
    except Exception:
        pass
    return None


def budget_bytes(device: str = "ram", *, fraction: float = 0.2) -> Optional[int]:
    """Return the spendable byte budget for ``device`` (``"ram"``/``"vram"``).

    ``fraction`` of the available memory, capped by the corresponding
    ``WORLDSIM_*_BUDGET_MB`` override.  ``None`` when nothing can be probed.
    """
    fraction = _env_fraction() or fraction
    if device == "vram":
        avail = available_vram_bytes()
        cap = _env_mb("WORLDSIM_VRAM_BUDGET_MB")
    else:
        avail = available_ram_bytes()
        cap = _env_mb("WORLDSIM_RAM_BUDGET_MB")
    if avail is None:
        return cap
    budget = int(avail * fraction)
    if cap is not None:
        budget = min(budget, cap)
    return budget


def auto_batch_size(
    item_nbytes: int,
    *,
    device: str = "ram",
    fraction: float = 0.2,
    lo: int = 1,
    hi: Optional[int] = None,
    default: Optional[int] = None,
) -> int:
    """Return how many ``item_nbytes``-sized items fit in the memory budget.

    Clamped to ``[lo, hi]``.  When the budget cannot be probed the caller's
    ``default`` (or ``hi``, or ``lo``) is returned so constants-era behaviour
    is preserved on exotic platforms.
    """
    budget = budget_bytes(device, fraction=fraction)
    if budget is None:
        size = default if default is not None else (hi if hi is not None else lo)
    else:
        size = budget // max(1, item_nbytes)
    size = max(lo, int(size))
    if hi is not None:
        size = min(size, hi)
    return size


def iter_memory_batches(
    items: Sequence,
    item_nbytes: int,
    *,
    device: str = "ram",
    fraction: float = 0.2,
    lo: int = 1,
    hi: Optional[int] = None,
    default: Optional[int] = None,
) -> Iterator[Sequence]:
    """Yield slices of ``items`` sized by :func:`auto_batch_size`.

    A single slice covering everything is yielded when the whole sequence
    fits the budget, so the common case adds no overhead.
    """
    n = len(items)
    if n == 0:
        return
    chunk = auto_batch_size(
        item_nbytes, device=device, fraction=fraction, lo=lo, hi=hi, default=default
    )
    if chunk >= n:
        yield items
        return
    for start in range(0, n, chunk):
        yield items[start : start + chunk]
