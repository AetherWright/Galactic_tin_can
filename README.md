# Galactic Tin Can

A galaxy-scale nation simulation driven by reinforcement learning, genetic algorithms,
and a realistic demographic model.  Nations build cities, wage wars, research technology,
form alliances, and colonise star systems across a procedurally generated universe.

## Architecture overview

The simulation is decomposed into *arms* — one package per domain:

```
worldsim/
├── core/                — foundations shared by every arm
│   ├── backend.py       — NumPy/CuPy array backend selection
│   ├── native.py        — optional Rust/C++ accelerator loading
│   ├── flags.py         — runtime toggles (VERBOSE, APPROXIMATE, watch)
│   ├── geometry.py      — distance, travel time, polygon helpers
│   ├── growth.py        — logistic growth (demographic workhorse)
│   ├── memory.py        — RAM/VRAM probing + memory-budgeted batch sizing
│   ├── parallel.py      — shared process/thread pools
│   ├── routing.py       — weighted route graphs (Rust-accelerated)
│   └── timing.py        — wall-clock guards
├── ai/                  — every learning component
│   ├── perceptron.py    — SimplePerceptron (vectorised on NumPy/CuPy)
│   ├── networks.py      — sparse bidirectional LayeredNetwork (NumPy/CuPy)
│   ├── nelder_mead.py   — simplex minimiser
│   ├── policy.py        — NelderMeadPolicy base class
│   ├── roles.py         — per-role facades (WarAI, DiplomacyAI, …)
│   ├── strategic.py     — StrategicNelderMeadAI planner
│   ├── rnn/             — GRU base models + per-nation LoRA adapters
│   │   ├── base.py      — shared backbones, registry, replay training
│   │   ├── controller.py— RNNController (Double DQN + MetaGA + dynamic LR warmup)
│   │   └── roles.py     — Torch role wrappers (war, civilian overseer/dept, project,
│   │                      diplomacy, research, doctrine, fleet)
│   ├── native.py        — Rust/C++ perceptron banks for division controllers
│   ├── neat.py/graph.py — NEAT topology evolution
│   ├── meta_ga.py       — reward-weight genetic algorithm
│   ├── embeddings.py    — action embeddings
│   ├── representations.py — galactic state tensors for controllers
│   └── persistence.py   — cross-run model merging and seeding
├── galaxy/              — astrography
│   ├── star.py          — Star model + STARS registry
│   ├── genesis.py       — star-cluster generation, init_world
│   └── settling.py      — initial nation placement
├── planets/             — planetary surfaces and population
│   ├── planet.py        — planet state and settlement pipeline
│   ├── terrain.py       — biome definitions and resource tables
│   ├── surface/         — noise, heightmaps, resource/atmosphere maps, storms
│   ├── mining.py        — mine aging, terrain deformation, depletion
│   ├── demographics/    — city / colony / county population models
│   └── buildings.py     — physical structures on planet surfaces
├── society/             — culture, ideas, leaders, goals
├── nations/             — nation state and internal development
│   ├── nation.py        — Nation dataclass, process_turn() pipeline, upkeep
│   ├── economy.py       — log-scale funds, resource stockpiles, treasury/reserves
│   ├── government.py    — government forms and approval
│   ├── projects.py      — national project catalogue
│   ├── construction.py  — resource collection and build actions
│   └── civilian.py      — hierarchical civilian AI: overseer + 6 department models
├── military/            — armies, fleets and doctrine
│   ├── divisions.py     — ground divisions and recruitment
│   ├── logistics.py     — supply, readiness, attrition, stacking
│   ├── combat.py        — order issuing and the ground-war loop
│   ├── nuclear.py       — warhead production and strikes
│   ├── fleets.py        — space fleets and space combat
│   ├── ftl.py           — FTL drive tiers
│   ├── doctrine.py      — DoctrineSignal vocabulary + DoctrineAI
│   └── command.py       — fleet FSM and the issue_doctrine pipeline
├── diplomacy/           — messages, relations, war goals, peace, trade
├── research/            — three-subsystem technology architecture
│   ├── technology.py    — graph primitives + procedural tech generation
│   ├── effects.py       — unlock-effect callables
│   ├── subsystems.py    — physics / engineering / biology domains
│   └── director.py      — point allocation across subsystems
├── events/              — procedural events with Q-learned responses
│   └── rust_bridge.py   — opt-in rust_events crate backend
├── engine/              — the century loop
│   ├── loop.py          — run_simulation / SimulationLoop
│   ├── scoring.py       — per-century MetaGA fitness
│   ├── reporting.py     — console summaries
│   ├── territory.py     — planet upkeep, star ownership
│   ├── blocs.py         — alliance blocs, border pressure
│   ├── politics.py      — collapses, succession, civil wars
│   └── filters.py       — great filters, zombie culling
├── interface/           — console presentation
├── config/              — data files and loaders
└── native/              — Rust crates and C++ helpers
```

## Quick start

```bash
python -m worldsim            # run with default settings
python main.py --turns 100    # 100 centuries
```

Environment variables:

| Variable | Default | Description |
|---|---|---|
| `WORLDSIM_USE_MP` | `1` | Enable multiprocessing (set `0` to disable) |
| `WORLDSIM_NUM_WORKERS` | cpu_count | Worker count for process/thread pools |
| `WORLDSIM_USE_RUST` | `auto` | Enable Rust helpers when `cargo` is available |
| `WORLDSIM_AI_BACKEND` | `auto` | Array backend for the Nelder-Mead AI stack: `auto` (CuPy → NumPy → Python), `cupy`, `numpy`, `python` |
| `WORLDSIM_USE_RUST_EVENTS` | `0` | Route event decisions through the `rust_events` crate (also `--rust-events`) |
| `WORLDSIM_RAM_BUDGET_MB` | unset | Absolute cap on host-RAM batch budgets |
| `WORLDSIM_VRAM_BUDGET_MB` | unset | Absolute cap on GPU batch budgets |
| `WORLDSIM_MEM_FRACTION` | per-call | Override the fraction of free memory a batch may claim |

## GPU and parallelism

### Array backend

All numerical computation selects the fastest available backend at import time:

1. **CuPy** (NVIDIA CUDA) — heightmap generation, city/colony batch processing,
   `logistic_growth` on arrays, geometry helpers, and the Nelder-Mead AI stack
   (`SimplePerceptron` / `LayeredNetwork` run the same code vectorised on GPU)
2. **NumPy** — CPU fallback with identical API
3. **Rust / C++** — scalar hot-paths for logistic growth, polygon area/centroid,
   distance; one-vs-rest perceptron banks (`rust_ai`, `cpp_ai`) as the
   legacy per-division posture/movement fallback when torch is unavailable
   (see **Ground divisions** below for the torch-backed batched network);
   the `rust_events` crate as an opt-in event Q-table engine (`--rust-events`)

### Memory-aware batching

Batch-shaped hot paths size their chunks from the memory actually available
(`core/memory.py`) instead of fixed constants:

* **RNN replay buffers** cap their capacity against host RAM (a war-role
  experience is ~40 KB; the 8 000-entry ceiling is only used when RAM allows).
* **Base-model training batches** are sized against free VRAM on CUDA (free
  RAM on CPU), between 16 and 256 samples per step.
* **City/colony demographic updates** and the **star-ownership fan-out**
  split very large inputs into RAM-budgeted chunks.

Budgets degrade to the previous fixed constants when probes are unavailable,
and can be capped with the `WORLDSIM_*_BUDGET_MB` variables.

### PyTorch models (nation AI)

All GRU base models, LoRA adapters, and fleet controllers are moved to the
best available device at module load:

```python
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

All tensor allocation — hidden states, LoRA parameters, training sequences,
target networks — uses `device=_DEVICE` throughout.  Checkpoints load via
`map_location=_DEVICE` so saves are portable between CPU and GPU environments.

### Multiprocessing

Multiprocessing is **on by default** and sizes itself to the system CPU count:

```python
_CPU_COUNT  = multiprocessing.cpu_count()   # e.g. 12
_NUM_WORKERS = int(os.getenv("WORLDSIM_NUM_WORKERS", str(_CPU_COUNT)))
```

Nation `process_turn` and `finalize_turn` calls both run through a thread pool
(releasing the GIL for NumPy/CuPy operations) to parallelise per-nation AI and
city batch work across all logical CPUs.  Planet turns and star ownership
calculations use a separate process pool for CPU-bound work that bypasses the GIL.

### Batch processing performance

Measured on a 12-core machine with CUDA GPU, CuPy backend active,
`WORLDSIM_USE_MP=0` (single-threaded, no pool overhead):

| City count | 200 turns | Per-turn latency |
|---|---|---|
| 10 cities | 90 ms | ~450 µs |
| 50 cities | 95 ms | ~476 µs |
| 100 cities | 99 ms | ~493 µs |

City count has minimal impact on per-turn latency thanks to fully vectorised
NumPy/CuPy array operations — the bottleneck is Python overhead, not arithmetic.

## Population model

The simulation uses a **birth-rate / death-rate** demographic model in place of
a single logistic growth rate.  Each component responds independently to game
events, making population trajectories meaningful and strategically interesting.

### Birth rate

```
birth_rate = (BASE_BIRTH + frontier_bonus) × stability_mod × food_birth_mod × growth_mult
```

| Parameter | Value | Notes |
|---|---|---|
| `BASE_BIRTH` | 0.025 / turn | ~1.25 % per 20-year period |
| `frontier_bonus` | 0–0.015 | Pioneers at low density; fades as city fills |
| `stability_mod` | 0.5–1.0 | Civil war halves effective birth rate |
| `food_birth_mod` | 0.3–1.0 | Famine floors births; surplus maximises them |
| `growth_mult` | tech bonus | Pharmacology biology tech |

### Death rate

```
total_mort = natural + disease + radiation + starvation + overcrowding
```

| Component | Formula | Notes |
|---|---|---|
| Natural | `0.010 × healthcare_factor` | Hospitals reduce baseline mortality |
| Disease | `plague^1.5 × 0.15 × (1−resist) × healthcare_factor` | Super-linear; high plague causes collapse |
| Radiation | `radiation × 0.08` | Flat; independent of healthcare |
| Starvation | `max(0, (0.5−food_ratio) × 0.10 / 0.5)` | Kicks in below 50 % food adequacy |
| Overcrowding | `max(0, (density−0.90) × 0.04)` | Squalor above 90 % of carrying capacity |

`healthcare_factor = 1 / (1 + hospital_count × 0.15)` — each hospital
produces diminishing returns, compressing both natural and disease mortality.

### Infrastructure dynamics

Infrastructure adjusts toward an economic equilibrium rather than changing
randomly:

```
target_infra = min(50, econ_health × 20)
infra += (target_infra − infra) × 0.03   # 3 % of gap per turn
```

Where `econ_health = economy_linear / (city_count × 50)` — a fully-funded
nation (economy ≥ 50 per city) reaches `infra=20`; a collapsed economy
causes slow decay back toward `infra=1`.

### 50-turn population trajectories

Starting condition: 3 cities × 5 000 population = 15 000 total.

| Scenario | Final population | Avg infrastructure |
|---|---|---|
| Fertile planet, stable, wealthy | 59 823 | 14.5 |
| Desert planet, stable | 27 054 | 8.3 |
| Plague outbreak (level 0.5) | 2 976 | 8.3 |
| Plague (0.5) + 4 hospitals | 9 900 | 8.3 |
| Civil war (stability = 20) | 28 212 | 5.1 |
| Famine (food depleted) | 852 | 6.7 |

Key observations:
- Hospitals cut plague losses by ~3× (2 976 → 9 900).
- Civil war suppresses growth but does not cause outright decline on a
  food-secure planet; economic collapse is needed for that.
- Famine is the most catastrophic single factor — near-total collapse in 50 turns.
- Desert biomes converge to roughly half the population of fertile worlds over
  a century even with identical governance, reflecting real agricultural limits.

### Rural and colony dynamics

**Colonies** (pioneer settlements):
- Higher frontier birth rate: `BASE = 0.030`, `frontier_bonus` up to `+0.020`
- Higher base mortality (`0.014`) reflecting harsher conditions
- Capped at 8 000 population; civilian AI upgrades mature colonies to cities
- Food ratio derived per-colony from the host planet's stored food

**Rural population** (county level):
- Grows at `0.020` birth / `0.013` death (below urban rates; less healthcare)
- Slightly more resilient to food stress than urban centres
- Migrates into cities that are below 70 % of carrying capacity
  at rate `2.5 % × available_headroom` per turn
- Urban overflow (> 95 % density) spills back to rural pools

## AI architecture

Each nation runs a set of GRU-based role controllers: war, project, diplomacy,
research, doctrine, fleet, and a **hierarchical civilian** pair (overseer +
one model per department — see below), for 8 controllers total.  All
controllers share a single **base model** per role across all nations; each
nation adds a lightweight **LoRA adapter** that specialises the backbone
without touching its weights.

Training uses **Double DQN** with:
- Per-nation Adam optimiser for LoRA parameters, with an optional **dynamic
  learning-rate warmup** (used by the civilian overseer and department
  models): a freshly-constructed controller trains at up to 5× the base LR,
  decaying back to base over its first ~100 training steps, so newly-seeded
  nations pick up a workable policy quickly instead of learning at the same
  slow rate as a long-lived one.
- Shared replay buffer (8 000 transitions) per role for base model updates
- Target network hard-copied every 16 training steps
- MetaGA fitness modulates ε-greedy exploration and reward scaling

### Hierarchical civilian AI

The 21 civilian build actions are grouped into six departments — interior,
industry, defense, fleet, science, executive — each backed by its own model
(`civilian_{slug}`, sharing weights across nations the same way every other
role does). An **overseer** model (`civilian_overseer`) scores all six
departments against the current turn's state.

Departments don't take turns: every turn, each department whose actions are
currently valid gets to act once, **in the overseer's priority order** —
highest-scored department first. This means the overseer expresses a
*resource priority* (interior over industry this turn, say) rather than
picking one winner and starving the rest. Both the overseer and every
department model that acted are trained every turn, so — unlike a
single-router design — no department goes without training data just because
it wasn't picked.

Turn-by-turn spending is bounded by a **soft budget** (see below): once the
budget for the turn is exhausted, remaining lower-priority departments stand
down, though the single highest-priority action is always allowed so a
nation with no money isn't fully paralysed.

### Ground divisions

Divisions don't fit the per-role GRU pattern above — they're numerous,
short-lived, and stateless from one fifth to the next, so they get their own
**shared, stateless, batched** feedforward network (`worldsim.ai.rnn.divisions`)
instead: one forward call scores every division a nation owns at once, and
one batched backward pass trains every transition from that fifth in a
single step, rather than one perceptron update per division. Per-nation LoRA
specialises the shared trunk the same way as every other role; the legacy
Rust/C++/NumPy perceptron pair (posture + movement) is the fallback when
torch is unavailable.

Each division picks one of nine actions per fifth per enemy it faces:
`attack`, `defend`, `hold_reserve`, `fortify`, `garrison_colony`, `raid`,
`besiege`, `deploy_offworld`, `recall_home` — up from the original
attack/defend/reserve (+ stay/deploy/recall) split across two separate
perceptrons. Training combines the immediate combat outcome with a
**colonial-capability / interstellar-force-projection bonus**: reward grows
with the nation's owned star count, the fraction of its soldiers stationed
off the homeworld, and how many divisions are actively garrisoning a colony
(`worldsim.military.divisions.compute_division_reward`).

### Soft economic budget

`Economy` tracks a **treasury** (`reserves`) alongside the existing log-scaled
`funds` gauge. Each turn, `settle_turn()` banks that turn's net income (gross
output minus a lightweight per-turn **upkeep** charge for every standing
building, division, and ship) into the treasury, floored at 0 and capped at
5× gross output so it can't grow unbounded. The civilian AI's per-turn
construction budget is net income plus a fixed slice of the treasury —
richer, more established nations can fund several departments' worth of
construction in one turn; poor or newly-founded ones are throttled back to
their single top priority.

## Requirements

- Python 3.12+
- `torch` ≥ 2.0 (CPU or CUDA)
- `numpy`
- `cupy` (optional, for GPU array acceleration)
- `yaml`, `Pillow`

Optional native accelerators compiled automatically on first run if tools are present:
- Rust (`cargo`) — logistic growth batches, distance, polygon helpers
- GCC — C++ fallback for the same scalar hot-paths
