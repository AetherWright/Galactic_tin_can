"""Command line interface for WorldSim.

Example
-------
Run a short game with five nations for fifty turns::

    python -m worldsim --nations 5 --turns 50

Use ``--profile`` to profile execution or ``--fast`` for approximate logic.
"""

import argparse
import os
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    """Run a WorldSim simulation from the command line."""
    parser = argparse.ArgumentParser(description="Run a WorldSim simulation")
    parser.add_argument("--nations", type=int, default=5, help="number of nations")
    parser.add_argument("--turns", type=int, default=50, help="simulation turns to run (1 turn ≈ 20 sim years)")
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="wall-clock limit to avoid hangs",
    )
    parser.add_argument(
        "--timeout-per-turn",
        action="store_true",
        help="apply --max-seconds to each turn instead of the overall run",
    )
    parser.add_argument(
        "--infinite",
        action="store_true",
        help="run without a turn limit (Ctrl+C to stop)",
    )
    parser.add_argument("--stars", type=int, default=3, help="number of star systems")
    parser.add_argument(
        "--population", type=int, default=5000, help="starting population per nation"
    )
    parser.add_argument("--neat", action="store_true", help="use NEAT-based reinforcement models")
    parser.add_argument(
        "--no-rust-events",
        action="store_true",
        help="disable Rust event helper",
    )
    parser.add_argument(
        "--qtable-path",
        default=str(Path.cwd() / "events_q_table.yml"),
        help="path to persistent event Q-table",
    )
    parser.add_argument(
        "--no-rust",
        action="store_true",
        help="disable Rust acceleration",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress detailed output")
    parser.add_argument("--watch", default=None, metavar="NAME",
                        help="nation name to show detailed output for")
    parser.add_argument("--profile", action="store_true", help="profile the simulation")
    parser.add_argument(
        "--profile-output",
        default=None,
        help="save detailed profiling stats to this file",
    )
    parser.add_argument("--fast", action="store_true", help="use approximations for speed")
    parser.add_argument(
        "--console",
        action="store_true",
        help="show per-nation console windows (requires curses; install "
        "windows-curses on Windows)",
    )
    parser.add_argument(
        "--json-log",
        default=str(Path.cwd() / "sim_log.jsonl"),
        help="save raw event data to this file",
    )
    parser.add_argument(
        "--ai-seed-dir",
        default=str(Path.cwd() / "worldsim_ai_seed"),
        metavar="DIR",
        help="directory for loading and saving merged AI model seeds across runs "
             "(default: worldsim_ai_seed/ in current directory)",
    )
    parser.add_argument(
        "--no-seed",
        action="store_true",
        help="disable loading existing AI seeds at start and saving seeds at end",
    )
    args = parser.parse_args(argv)


    from .initialization import init_world
    from .simulation import run_simulation
    from .ai import set_use_neat, set_use_rust as set_rust_ai
    from . import utils
    from .utils import set_use_rust as set_rust_utils
    from .routes import set_use_rust as set_rust_graph
    from .save import merge_and_save_models, load_model_seeds

    set_use_neat(args.neat)
    utils.VERBOSE    = not args.quiet
    utils.APPROXIMATE = args.fast
    if args.no_rust:
        set_rust_ai(False)
        set_rust_utils(False)
        set_rust_graph(False)
    nations = init_world(
        args.nations,
        args.stars,
        starting_population=args.population,
    )

    # Inject AI seeds from the previous run so models start smarter.
    seed_dir = Path(args.ai_seed_dir)
    if not args.no_seed and seed_dir.exists():
        loaded = load_model_seeds(nations, seed_dir)
        if loaded and not args.quiet:
            print(f"[worldsim] Loaded AI seeds from {seed_dir}")

    turns = None if args.infinite else args.turns
    if args.profile:
        import cProfile
        import pstats
        prof = cProfile.Profile()
        prof.enable()
        run_simulation(
            nations,
            turns=turns,
            max_seconds=args.max_seconds,
            per_turn_timeout=args.timeout_per_turn,
            use_neat=args.neat,
            log_path=args.json_log,
            use_rust_events=False if args.no_rust_events else None,
            qtable_path=args.qtable_path,
            console=args.console,
        )
        prof.disable()
        stats = pstats.Stats(prof).sort_stats(pstats.SortKey.CUMULATIVE)
        stats.print_stats(20)
        if args.profile_output:
            stats.dump_stats(args.profile_output)
    else:
        run_simulation(
            nations,
            turns=turns,
            max_seconds=args.max_seconds,
            per_turn_timeout=args.timeout_per_turn,
            use_neat=args.neat,
            log_path=args.json_log,
            use_rust_events=False if args.no_rust_events else None,
            qtable_path=args.qtable_path,
            console=args.console,
            watch_nation=args.watch,
        )

    # Persist merged AI seeds for the next run.
    if not args.no_seed:
        try:
            merge_and_save_models(nations, seed_dir)
            if not args.quiet:
                print(f"[worldsim] Saved AI seeds to {seed_dir}")
        except Exception as exc:
            if not args.quiet:
                print(f"[worldsim] Warning: could not save AI seeds: {exc}")


if __name__ == "__main__":
    main()
