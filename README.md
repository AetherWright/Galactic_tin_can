# Galactic Tin Can

A galaxy-scale nation simulation driven by reinforcement learning, genetic algorithms,
and a realistic demographic model.  Nations build cities, wage wars, research technology,
form alliances, and colonise star systems across a procedurally generated universe.

## Architecture overview

```
worldsim/
├── simulation.py        — main century loop (run_simulation / SimulationLoop)
├── models/
│   ├── nation.py        — nation state, process_turn(), AI wiring
│   ├── economy.py       — log-scale funds + resource stockpiles
│   ├── military_ai.py   — fleet FSM and doctrine controllers
│   ├── war.py           — combat resolution, nuclear strikes
│   ├── diplomacy.py     — alliances, trade, tribute, messages
│   └── tech.py          — technology tree with unlock effects
├── planets/
│   ├── city.py          — urban demographic model (birth/death)
│   ├── colony.py        — pioneer settlement model
│   ├── county.py        — rural population dynamics
│   ├── planet.py        — planet state, surface generation
│   └── heightmap.py     — fBm terrain + resource heatmaps
├── torch_rnn.py         — GRU base models + per-nation LoRA adapters
├── torch_ai.py          — MLP fleet controller
├── cupy_ai.py           — CuPy-accelerated perceptron / layered network
├── mp_utils.py          — multiprocessing and thread pool helpers
└── utils.py             — GPU/NumPy backend selection, logistic growth
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

## GPU and parallelism

### Array backend

All numerical computation selects the fastest available backend at import time:

1. **CuPy** (NVIDIA CUDA) — heightmap generation, city/colony batch processing,
   `logistic_growth` on arrays, geometry helpers
2. **NumPy** — CPU fallback with identical API
3. **Rust / C++** — scalar hot-paths for logistic growth, polygon area/centroid,
   distance

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

Each nation runs seven GRU-based role controllers (war, civilian, project,
diplomacy, research, doctrine, fleet).  All controllers share a single
**base model** per role across all nations; each nation adds a lightweight
**LoRA adapter** that specialises the backbone without touching its weights.

Training uses **Double DQN** with:
- Per-nation Adam optimiser for LoRA parameters
- Shared replay buffer (8 000 transitions) per role for base model updates
- Target network hard-copied every 16 training steps
- MetaGA fitness modulates ε-greedy exploration and reward scaling

## Requirements

- Python 3.12+
- `torch` ≥ 2.0 (CPU or CUDA)
- `numpy`
- `cupy` (optional, for GPU array acceleration)
- `yaml`, `Pillow`

Optional native accelerators compiled automatically on first run if tools are present:
- Rust (`cargo`) — logistic growth batches, distance, polygon helpers
- GCC — C++ fallback for the same scalar hot-paths
