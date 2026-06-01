# DRL Project — Adapted Loss Function with Symlog

> **Course**: Deep Reinforcement Learning  
> **Task**: Adapted Loss Function — Imagination-based model → Symlog activation function

## Quick Start

```bash
# 1. Create virtual environment (already done if .venv exists)
python -m venv .venv

# 2. Install dependencies with CUDA support (RTX 4050)
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu124
.venv\Scripts\pip install gymnasium[classic_control] numpy matplotlib

# 3. Run a single training experiment
.venv\Scripts\python -m src.train --agent imagination --env CartPole-v1 --use-symlog --seed 0

# 4. Run the full experiment suite (30 runs: 3 conditions × 5 seeds × 2 envs)
.venv\Scripts\python -m src.run_experiments

# 5. Generate comparison plots from results
.venv\Scripts\python -m src.evaluate --experiment-dir experiments
```

## Project Overview

This project investigates whether applying **symlog-transformed losses** in an imagination-based reinforcement learning agent improves training stability and performance compared to standard MSE losses.

**Symlog** (`sign(x) · ln(|x| + 1)`) compresses large-magnitude values while preserving near-zero distinctions, making gradient magnitudes stable regardless of reward scale. It was introduced in **DreamerV3** (Hafner et al., 2023).

We compare three experimental conditions:

| Condition | Agent | World Model | Loss Type |
|-----------|-------|-------------|-----------|
| **A** — Model-Free AC | Baseline | ✗ | Standard MSE |
| **B** — Imagination AC | Dyna-style | ✓ | Standard MSE |
| **C** — Imagination AC + Symlog | Dyna-style | ✓ | **Symlog MSE** |

## Project Structure

```
DRL-Project/
├── docs/                        # Documentation
│   ├── README.md                # ← This file
│   ├── architecture.md          # Detailed architecture documentation
│   ├── project_scope.md         # Full project scope and plan
│   ├── decisions.md             # Technical decisions log
│   └── brainstorm.md            # Original assignment notes
├── report/                      # LaTeX report template
├── src/                         # Source code
│   ├── config.py                # Centralized configuration (Config dataclass)
│   ├── train.py                 # Training entry point (CLI)
│   ├── evaluate.py              # Post-experiment analysis and plotting
│   ├── run_experiments.py       # Automation: runs full experiment grid
│   ├── models/                  # Neural networks
│   │   ├── networks.py          #   Shared MLP builder
│   │   ├── world_model.py       #   Dynamics + reward prediction
│   │   ├── actor.py             #   Categorical policy network
│   │   └── critic.py            #   Value network (symlog-aware)
│   ├── agents/                  # Agent implementations
│   │   ├── base.py              #   Abstract base class
│   │   ├── model_free_ac.py     #   Condition A
│   │   └── imagination_ac.py    #   Conditions B & C
│   ├── losses/                  # Loss functions
│   │   └── symlog.py            #   symlog, symexp, symlog_mse_loss
│   └── utils/                   # Utilities
│       ├── replay_buffer.py     #   Experience replay
│       ├── logger.py            #   Metrics logging (JSON)
│       └── plotting.py          #   Matplotlib visualizations
├── experiments/                 # Generated: experiment outputs
│   ├── {experiment_name}/       #   Per-run: config.json, metrics.json, agent.pt
│   └── plots/                   #   Generated comparison plots
├── requirements.txt
└── .venv/                       # Virtual environment
```

## CLI Reference

### `src.train` — Train a single agent

```bash
python -m src.train [OPTIONS]

Options:
  --agent {model_free,imagination}  Agent type (default: imagination)
  --env ENV_NAME                    Gymnasium environment (default: CartPole-v1)
  --use-symlog                      Enable symlog loss adaptation
  --seed SEED                       Random seed (default: 0)
  --total-steps N                   Total environment steps (default: 100000)
  --experiment-dir DIR              Output directory (default: experiments)
```

### `src.run_experiments` — Run full experiment grid

```bash
python -m src.run_experiments [OPTIONS]

Options:
  --experiment-dir DIR    Output directory (default: experiments)
  --dry-run               Preview commands without executing
```

Runs: 2 envs × 3 conditions × 5 seeds = **30 experiments**.

### `src.evaluate` — Aggregate results and generate plots

```bash
python -m src.evaluate [OPTIONS]

Options:
  --experiment-dir DIR    Directory with experiment results (default: experiments)
```

Generates: learning curves, loss comparisons, world model error plots.

## Hardware Requirements

- **Minimum**: CPU with 8GB RAM (~5 hours for full suite)
- **Recommended**: NVIDIA GPU with 4+ GB VRAM (~1-2 hours for full suite)
- **Tested on**: RTX 4050 Laptop (6GB VRAM), CUDA 13.2

The code auto-detects CUDA and uses GPU if available (`device: "auto"` in Config).
