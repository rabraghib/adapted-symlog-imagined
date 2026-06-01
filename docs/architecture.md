# Architecture Documentation

## Overview

This project implements a **simplified Dreamer-like imagination-based agent** for
discrete-action Gymnasium environments. The core idea: learn a world model from
real experience, then use it to *imagine* future trajectories and train the policy
on those imagined rollouts.

Unlike full DreamerV3, we operate in **raw state space** (not learned latent states)
and use a **Dyna-style hybrid** approach (train on both real and imagined data).

---

## Component Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                     REAL ENVIRONMENT                         │
│                (CartPole-v1 / LunarLander-v3)                │
└────────────┬─────────────────────────────┬───────────────────┘
             │ (state, action, reward,     │ action
             │  next_state, done)          │
             ▼                             │
┌──────────────────────────────────┐       │
│         REPLAY BUFFER            │       │
│  (pre-allocated NumPy arrays,    │       │
│   capacity: 100K, zero-copy      │       │
│   torch.from_numpy() sampling)   │       │
└──────┬───────────────────────────┘       │
       │ sample batches                    │
       ▼                                   │
┌────────────────────────┐                 │
│     WORLD MODEL        │                 │
│  Dynamics: (s,a)→Δs    │                 │
│  Reward:   (s,a)→r     │                 │
│  (optional symlog)     │                 │
└──────┬─────────────────┘                 │
       │ imagined rollouts                 │
       ▼                                   │
┌────────────────────────────────────┐     │
│        ACTOR-CRITIC                │     │
│  Actor:  π(a|s) — Categorical      ├─────┘
│  Critic: V(s)   — symlog-aware     │
│                                    │
│  Trained on:                       │
│   • Real transitions (every step)  │
│   • Imagined data (every Nth step) │
└────────────────────────────────────┘
```

---

## Neural Networks

All networks are simple MLPs built with `build_mlp()` from `src/models/networks.py`.
Default architecture: **2 hidden layers of 256 units** with ReLU activations.

### World Model (`src/models/world_model.py`)

The world model has two heads:

1. **Dynamics model**: Predicts state *residual* (`Δs`) — `next_state = state + Δs`
   - Input: `[state, one_hot(action)]` → Output: `state_dim`
   - Residual prediction improves stability (the model only needs to learn the *change*)
   - Loss: Standard MSE against true `next_state`

2. **Reward model**: Predicts scalar reward
   - Input: `[state, one_hot(action)]` → Output: `1`
   - Loss: Standard MSE **or** Symlog MSE (depending on `use_symlog`)
   - If symlog: network outputs in symlog space, `symexp()` applied at inference

**`imagine(start_states, actor, horizon)`**:
- Rolls out the world model for `horizon` steps using the actor's policy
- Returns lists of states, actions, rewards, log_probs, and final_states
- World model predictions use `torch.no_grad()` (no WM gradient during imagination)
- Actor gradients flow through `log_prob` for policy gradient computation

### Actor (`src/models/actor.py`)

- **Input**: state vector → **Output**: `Categorical` distribution over actions
- Methods:
  - `get_action(state, deterministic)` — sample or argmax
  - `evaluate(state, action)` — returns log_prob and entropy (for training)

### Critic (`src/models/critic.py`)

- **Input**: state vector → **Output**: scalar value estimate
- **Symlog-aware**: when `use_symlog=True`:
  - `forward()` returns value in **symlog space** (raw network output)
  - `get_value()` applies `symexp()` to return value in **original scale**
  - `compute_loss()` applies `symlog()` to targets before MSE comparison

---

## Agent Architectures

### Model-Free AC (`src/agents/model_free_ac.py`) — Condition A

Standard advantage actor-critic:
1. Sample batch from replay buffer
2. Compute TD(0) targets: `G = r + γ·V(s')·(1-done)`
3. Update critic: MSE(V(s), G)
4. Update actor: `-log_prob(a) · advantage - entropy_coef · H(π)`

No world model, no imagination. Serves as baseline.

### Imagination AC (`src/agents/imagination_ac.py`) — Conditions B & C

Hybrid Dyna-style agent with three training phases per step:

```
train_step(buffer):
    ┌─────────────────────────────────────────────────┐
    │ Phase 1A: Train World Model on real data        │ (every step)
    │   • Sample batch from buffer                    │
    │   • 2 gradient steps (world_model_train_steps)  │
    │   • Gradient clipping (norm=1.0)                │
    ├─────────────────────────────────────────────────┤
    │ Phase 1B: Train Actor-Critic on REAL data       │ (every step)
    │   • TD(0) targets from real transitions         │
    │   • Standard AC update (like model-free)        │
    │   • Keeps policy grounded in reality            │
    ├─────────────────────────────────────────────────┤
    │ Phase 2: Train Actor-Critic on IMAGINED data    │ (every 10th step)
    │   • Sample start states from buffer             │
    │   • Imagine 5-step rollouts through world model │
    │   • Compute n-step returns with critic bootstrap│
    │   • Update critic + actor on imagined data      │
    │   • Advantage normalization for stability       │
    └─────────────────────────────────────────────────┘
```

**Warmup phase** (first 1000 steps):
- Only Phase 1A runs (world model training)
- Agent uses **random actions** for diverse exploration
- After warmup: all three phases activate

**Why Dyna-style and not pure imagination?**
- Our simple MLP world model (operating in raw state space) has limited accuracy
- Pure imagination-based training caused policy collapse in testing
- Hybrid approach: real data keeps policy grounded, imagined data adds extra signal
- See `docs/decisions.md` Decision 6 for full rationale

---

## Symlog Integration

### Where symlog is applied (Condition C only)

| Component | Standard (B) | Symlog (C) |
|-----------|-------------|------------|
| Critic output | Raw value | Value in symlog space |
| Critic loss | `MSE(V(s), G)` | `MSE(V(s), symlog(G))` |
| Critic inference | `V(s)` directly | `symexp(V(s))` → original scale |
| WM reward output | Raw reward | Reward in symlog space |
| WM reward loss | `MSE(r̂, r)` | `MSE(r̂, symlog(r))` |
| WM reward inference | `r̂` directly | `symexp(r̂)` → original scale |

### Mathematical definitions

```python
symlog(x) = sign(x) · ln(|x| + 1)        # Compress large values
symexp(x) = sign(x) · (exp(|x|) - 1)     # Inverse (decompress)
```

### Numerical stability

- `symexp()` clamps input to `[-20, 20]` to prevent `exp()` overflow
- All optimizers use gradient clipping (max norm = 1.0)
- These safeguards prevent NaN propagation discovered during testing

---

## Hyperparameters

All hyperparameters are defined in `src/config.py` as a `Config` dataclass:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 3e-4 | Adam learning rate (shared by all networks) |
| `gamma` | 0.99 | Discount factor |
| `batch_size` | 64 | Mini-batch size |
| `buffer_size` | 100,000 | Replay buffer capacity |
| `hidden_dims` | (256, 256) | MLP hidden layer sizes |
| `entropy_coef` | 0.01 | Actor entropy bonus coefficient |
| `imagination_horizon` | 5 | Steps to imagine forward |
| `warmup_steps` | 1,000 | Steps before imagination begins |
| `world_model_train_steps` | 2 | WM gradient steps per train call |
| `imagination_train_ratio` | 10 | Imagination update every N steps |
| `num_imagine_batches` | 8 | Number of batches for imagination rollouts. **Defined in `config.py` but currently unused** — the agent uses `batch_size` directly for the number of start states. |
| `world_model_train_freq` | 1 | Train the world model every N environment steps. **Defined in `config.py` but currently unused** — the agent trains the world model on every step. |
| `grad_clip_norm` | 1.0 | Maximum gradient norm |
| `eval_freq` | 1,000 | Evaluate every N steps |
| `eval_episodes` | 10 | Episodes per evaluation |

### Per-environment step budgets

| Environment | Total Steps |
|-------------|-------------|
| CartPole-v1 | 100,000 |
| LunarLander-v3 | 300,000 |

---

## Experiment Grid

The full experiment suite (`src/run_experiments.py`) runs:

- **2 environments**: CartPole-v1, LunarLander-v3
- **3 conditions**: Model-Free AC, Imagination AC, Imagination AC + Symlog
- **5 seeds**: 0, 1, 2, 3, 4

**Total: 30 runs**

### Output structure

```
experiments/
├── CartPole-v1_model_free_seed0/
│   ├── config.json        # Run configuration
│   ├── metrics.json       # Training + eval metrics
│   └── agent.pt           # Saved model checkpoint
├── CartPole-v1_imagination_seed0/
├── CartPole-v1_imagination_symlog_seed0/
├── ...
└── plots/                 # Generated by evaluate.py
    ├── CartPole-v1_learning_curves.png
    ├── CartPole-v1_loss_comparison.png
    ├── CartPole-v1_world_model_error.png
    ├── LunarLander-v3_learning_curves.png
    └── ...
```

---

## Performance Optimizations for Parallel Execution

When running the full experiment grid with `--jobs N`, several optimizations
prevent CPU thread thrashing and improve throughput:

### Thread pinning

- **`torch.set_num_threads(1)`** is called at process startup in each spawned
  subprocess. This prevents PyTorch's internal thread pool from competing with
  other parallel workers.
- The following environment variables are set to `"1"` in every spawned
  subprocess to constrain underlying BLAS / math libraries to a single thread:
  - `OMP_NUM_THREADS`
  - `MKL_NUM_THREADS`
  - `OPENBLAS_NUM_THREADS`
  - `VECLIB_MAXIMUM_THREADS`
  - `NUMEXPR_NUM_THREADS`

### Vectorized replay buffer

The replay buffer (`src/utils/replay_buffer.py`) uses **pre-allocated contiguous
NumPy arrays** instead of a Python `deque`. Key properties:

- Memory is allocated once via `np.zeros()` at first insertion (lazy
  initialization based on observed `state_dim`).
- Sampling uses `np.random.randint()` for vectorized index generation — no
  Python-level loops or list comprehensions.
- Tensors are created with `torch.from_numpy()` for zero-copy sharing between
  NumPy and PyTorch (data is not duplicated in memory).

These optimizations ensure that parallel runs with `--jobs N` scale efficiently
without per-process CPU over-subscription.

---

## Key References

1. **Hafner et al. (2023)** — *Mastering Diverse Domains through World Models* (DreamerV3). NeurIPS 2023.
2. **Hafner et al. (2020)** — *Dream to Control: Learning Behaviors by Latent Imagination* (DreamerV1).
3. **Sutton (1991)** — *Dyna, an Integrated Architecture for Learning, Planning, and Reacting.*
4. **Mnih et al. (2015)** — *Human-level control through deep reinforcement learning* (DQN).
