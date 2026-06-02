import os
import multiprocessing as _mp
from multiprocessing.pool import Pool as _Pool
from typing import Iterable, Callable, Any, Literal
from concurrent.futures import ThreadPoolExecutor

# Enable multiprocessing by default; set WORLDSIM_USE_MP=0 to disable.
_USE_MP = bool(int(os.getenv("WORLDSIM_USE_MP", "1")))

# Number of worker processes/threads.  Defaults to the logical CPU count;
# override with WORLDSIM_NUM_WORKERS.
_CPU_COUNT = max(1, _mp.cpu_count() or 1)
_NUM_WORKERS = int(os.getenv("WORLDSIM_NUM_WORKERS", str(_CPU_COUNT)))

_POOL: _Pool | None = None
_THREAD_POOL: ThreadPoolExecutor | None = None


def _get_pool() -> _Pool | None:
    global _POOL
    if not _USE_MP:
        return None
    if _POOL is None:
        _POOL = _mp.Pool(processes=_NUM_WORKERS)
    return _POOL


def pooled_map(
    func: Callable[[Any], Any],
    iterable: Iterable[Any],
    *,
    mode: Literal["process", "thread", "sync"] = "process",
    chunksize: int = 0,
) -> list[Any]:
    """Apply ``func`` to each element of ``iterable`` using a shared pool.

    The helper unifies the previous ``mp_map`` and ``tp_map`` utilities so the
    worker selection logic lives in one place. ``mode`` controls the
    concurrency strategy and can be one of ``"process"``, ``"thread"`` or
    ``"sync"``. When multiprocessing is disabled the function falls back to
    synchronous execution automatically.
    """

    items = list(iterable)
    if not items:
        return []

    normalized_mode = mode
    if mode not in {"process", "thread", "sync"}:
        raise ValueError(f"Unsupported pooled_map mode: {mode!r}")

    if not _USE_MP or normalized_mode == "sync":
        return [func(x) for x in items]

    if normalized_mode == "process":
        pool = _get_pool()
        if pool is None:
            return [func(x) for x in items]
        effective_chunksize = chunksize
        if effective_chunksize <= 0:
            processes = max(pool._processes or 1, 1)
            effective_chunksize = max(1, len(items) // (processes * 4))
        return list(pool.map(func, items, effective_chunksize))

    # Thread mode -----------------------------------------------------
    global _THREAD_POOL
    if _THREAD_POOL is None:
        _THREAD_POOL = ThreadPoolExecutor(max_workers=_NUM_WORKERS)
    futures = [_THREAD_POOL.submit(func, x) for x in items]
    return [future.result() for future in futures]


def mp_map(func: Callable[[Any], Any], iterable: Iterable[Any], *, chunksize: int = 0) -> list[Any]:
    """Compatibility shim using :func:`pooled_map` in ``process`` mode."""

    return pooled_map(func, iterable, mode="process", chunksize=chunksize)


def shutdown_pool() -> None:
    """Terminate the shared pool if running."""
    global _POOL, _THREAD_POOL
    if _POOL is not None:
        _POOL.close()
        _POOL.join()
        _POOL = None
    if _THREAD_POOL is not None:
        _THREAD_POOL.shutdown()
        _THREAD_POOL = None


def tp_map(func: Callable[[Any], Any], iterable: Iterable[Any]) -> list[Any]:
    """Compatibility shim using :func:`pooled_map` in ``thread`` mode."""

    return pooled_map(func, iterable, mode="thread")


def planet_turn_worker(state: tuple[str, float, float]) -> tuple[str, float, float]:
    name, plague, radiation = state
    if plague > 0:
        plague = max(0.0, plague - 0.01)
    if radiation > 0:
        radiation = max(0.0, radiation - 0.005)
    return name, plague, radiation


def star_owner_worker(
    data: tuple[str, list, dict[int, float], dict[int, float]]
) -> tuple[str, int | None]:
    """Return the most influential nation within a star system.

    Cities still determine base control. National economies and
    military strength act as modifiers so wealthy or heavily armed
    nations can claim contested systems. Fleets are approximated by
    the ``military`` score for now.
    """

    name, planets_data, economies, militaries = data
    scores: dict[int, float] = {}
    for city_list in planets_data:
        pop_by_owner: dict[int, int] = {}
        for owner, pop in city_list:
            if owner is not None:
                pop_by_owner[owner] = pop_by_owner.get(owner, 0) + pop
        if pop_by_owner:
            owner = max(pop_by_owner.items(), key=lambda kv: kv[1])[0]
            econ_bonus = economies.get(owner, 0.0) / 1000.0
            mil_bonus = militaries.get(owner, 0.0) / 100.0
            scores[owner] = scores.get(owner, 0.0) + 1 + econ_bonus + mil_bonus
    new_owner = max(scores.items(), key=lambda kv: kv[1])[0] if scores else None
    return name, new_owner
