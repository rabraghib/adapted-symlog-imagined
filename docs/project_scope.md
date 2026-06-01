# Project Scope: Adapted Loss Function with Symlog in Imagination-Based DRL

## 1. Problem Statement

In Deep Reinforcement Learning, training stability and generalization across environments
with different reward scales remains a key challenge. Standard loss functions (e.g., MSE on
TD targets) are sensitive to the magnitude of rewards — large rewards cause exploding gradients,
while tiny rewards produce vanishing learning signals.

**DreamerV3** (Hafner et al., 2023) introduced the **symlog transformation** as a solution:
applying `symlog(x) = sign(x) · ln(|x| + 1)` to loss targets compresses extreme values
while preserving near-zero distinctions. This is used inside an **imagination-based**
architecture, where a learned world model generates imagined trajectories to train
the policy without interacting with the real environment.

Our project investigates: **Does applying symlog-transformed losses in an imagination-based
agent improve training stability and performance compared to standard losses?**

---

## 2. Project Objectives

1. Implement a **simplified imagination-based RL agent** (inspired by Dreamer) from scratch
   using PyTorch.
2. Implement the **symlog transformation** and integrate it into the agent's loss functions
   (critic loss, reward prediction loss).
3. Run controlled experiments comparing:
   - **Baseline A**: Standard Actor-Critic (model-free, no world model) — to show the value of imagination
   - **Baseline B**: Imagination-based Actor-Critic with standard MSE loss — to isolate symlog's effect
   - **Proposed**: Imagination-based Actor-Critic with symlog-adapted loss
4. Analyze and report results with learning curves, stability metrics, and ablation discussion.

---

## 3. Background Concepts

### 3.1 Symlog Transformation
```
symlog(x) = sign(x) · ln(|x| + 1)
symexp(x) = sign(x) · (exp(|x|) - 1)     # inverse
```

Properties:
- Compresses large magnitudes (both positive and negative)
- Near-linear around zero (preserves small reward distinctions)
- Symmetric: treats positive and negative values equally
- Smooth and differentiable everywhere

### 3.2 Imagination-Based Training (World Model)
Instead of learning only from real environment interactions:
1. **Learn a world model** from real experience: `(s, a) → (s', r)`
2. **Imagine** future trajectories by rolling out the world model
3. **Train actor and critic** on these imagined trajectories (no real env needed)
4. **Act** in the real environment using the trained policy, collect data, repeat

This is the core idea behind the Dreamer family (Dreamer, DreamerV2, DreamerV3).

### 3.3 Where Symlog Enters the Loss
In the standard setup:
- **Critic loss**: `L_critic = MSE(V(s), target_return)` — target can be huge or tiny
- **Reward prediction loss**: `L_reward = MSE(R_pred(s,a), r_actual)` — same problem

With symlog adaptation:
- **Critic loss**: `L_critic = MSE(V(s), symlog(target_return))` — compressed targets
- **Reward prediction loss**: `L_reward = MSE(R_pred(s,a), symlog(r_actual))` — compressed
- At inference: apply `symexp()` to recover original scale

This keeps gradient magnitudes stable regardless of environment reward scale.

---

## 4. Architecture Overview

We build a **simplified Dreamer-like agent** (NOT DreamerV3's full complexity — no latent
RSSM, no KL balancing, no two-hot encoding). Our simplification works in raw state space.

```
┌─────────────────────────────────────────────────────────┐
│                    REAL ENVIRONMENT                      │
│                  (CartPole / LunarLander)                │
└──────────────┬──────────────────────┬───────────────────┘
               │ (s, a, r, s')       │ action
               ▼                     │
┌──────────────────────────┐         │
│      REPLAY BUFFER       │         │
└──────────┬───────────────┘         │
           │ sample batches          │
           ▼                         │
┌──────────────────────────┐         │
│      WORLD MODEL         │         │
│  MLP: (s, a) → (s', r)  │         │
│  Trained on real data    │         │
└──────────┬───────────────┘         │
           │ imagined rollouts       │
           ▼                         │
┌──────────────────────────┐         │
│   ACTOR-CRITIC           │         │
│  Actor:  π(a|s)          ├─────────┘
│  Critic: V(s)            │
│  Trained on imagined     │
│  trajectories            │
│                          │
│  ★ SYMLOG applied here   │
│    to critic targets     │
│    and reward predictions│
└──────────────────────────┘
```

### Components (all MLPs):

| Component | Input | Output | Loss |
|-----------|-------|--------|------|
| **World Model — Dynamics** | (state, action) | next_state | MSE |
| **World Model — Reward** | (state, action) | reward | MSE or **symlog MSE** ★ |
| **Critic** | state | value estimate | MSE or **symlog MSE** ★ |
| **Actor** | state | action distribution | Policy gradient (using imagined returns) |

★ = where symlog adaptation is applied in the experimental condition

---

## 5. Experiment Plan

### 5.1 Environments

| Environment | Why | Reward characteristics |
|-------------|-----|----------------------|
| **CartPole-v1** | Simple, fast training (~minutes), easy to debug | Reward = +1 per step, max 500. Small scale. |
| **LunarLander-v3** | Moderately complex, continuous state, shaped reward | Reward range ~[-200, +300]. Mixed scale, crash penalties. |

Both use **Gymnasium** (successor to OpenAI Gym). Discrete action spaces for simplicity.

### 5.2 Experimental Conditions

| Condition | World Model | Actor-Critic training | Loss type |
|-----------|-------------|----------------------|-----------|
| **A — Model-Free AC** | ✗ | On real transitions | Standard MSE |
| **B — Imagination AC** | ✓ | Hybrid: real transitions + imagined rollouts (Dyna-style) | Standard MSE |
| **C — Imagination AC + Symlog** | ✓ | Hybrid: real transitions + imagined rollouts (Dyna-style) | Symlog MSE |

Each condition run with **5 random seeds** for statistical reliability.

### 5.3 Metrics

1. **Episodic return** (mean ± std over seeds) vs. training steps → learning curves
2. **Training stability**: variance of returns across seeds / across training
3. **World model prediction error**: how well does the model learn the dynamics?
4. **Wall-clock time**: practical training cost comparison

### 5.4 Optional Extension (if time permits)
- **Reward scaling experiment**: Multiply environment rewards by 100x or 0.01x, show that
  symlog-based loss is robust while standard MSE degrades. This directly demonstrates
  the motivation for symlog.

---

## 6. Technology Stack

| Tool | Purpose |
|------|---------|
| **Python 3.10+** | Language |
| **PyTorch** | Neural networks, autograd |
| **Gymnasium** | RL environments (CartPole-v1, LunarLander-v3) |
| **NumPy** | Numerical operations |
| **Matplotlib** | Plotting learning curves and comparisons |
| **TensorBoard** (optional) | Live training monitoring |

---

## 7. Project Structure

```
DRL-Project/
├── docs/                        # Documentation
│   ├── README.md                #   Quick start and project overview
│   ├── architecture.md          #   Detailed architecture documentation
│   ├── project_scope.md         #   ← This file
│   ├── decisions.md             #   Technical decisions log (8 decisions)
│   ├── brainstorm.md            #   Original assignment notes
│   └── results_analysis.md      #   Analysis of experiment graphs
├── report/                      # LaTeX report template
├── src/                         # Source code
│   ├── config.py                #   Config dataclass (all hyperparameters)
│   ├── train.py                 #   Training entry point (CLI)
│   ├── evaluate.py              #   Post-experiment analysis and plotting
│   ├── run_experiments.py       #   Automation: full experiment grid
│   ├── models/                  #   Neural networks
│   │   ├── networks.py          #     Shared MLP builder
│   │   ├── world_model.py       #     Dynamics + reward predictor
│   │   ├── actor.py             #     Categorical policy
│   │   └── critic.py            #     Value network (symlog-aware)
│   ├── agents/                  #   Agent implementations
│   │   ├── base.py              #     Abstract base class
│   │   ├── model_free_ac.py     #     Condition A (baseline)
│   │   └── imagination_ac.py    #     Conditions B & C (Dyna-style)
│   ├── losses/                  #   Loss functions
│   │   └── symlog.py            #     symlog, symexp, symlog_mse_loss
│   └── utils/                   #   Utilities
│       ├── replay_buffer.py     #     Experience replay buffer
│       ├── logger.py            #     JSON metrics logging
│       └── plotting.py          #     Matplotlib visualizations
├── experiments/                 # Generated: experiment results + plots
├── requirements.txt             # Includes gymnasium[box2d] for LunarLander
└── .venv/                       # Virtual environment
```

---

## 8. Timeline (2 weeks)

### Week 1: Build & Debug
| Day | Task |
|-----|------|
| Day 1-2 | Project setup, env wrappers, replay buffer, utility code |
| Day 3 | Implement world model (dynamics + reward MLP) |
| Day 4 | Implement actor-critic + model-free baseline (Condition A) |
| Day 5 | Implement imagination-based training loop (Condition B) |
| Day 6 | Implement symlog loss + integrate (Condition C) |
| Day 7 | Debug & sanity checks — all three conditions train and converge on CartPole |

### Week 2: Experiment & Write
| Day | Task |
|-----|------|
| Day 8-9 | Run full experiments: both envs × 3 conditions × 5 seeds |
| Day 10 | Generate plots, collect metrics, analyze results |
| Day 11-12 | Write report (chapters: intro, lit review, methodology, results, discussion) |
| Day 13 | Report polish, add figures, proofread |
| Day 14 | Buffer day / final submission |

---

## 9. Key References

1. **Hafner et al. (2023)** — "Mastering Diverse Domains through World Models" (DreamerV3).
   NeurIPS 2023. *The paper that introduced symlog for DRL.*
2. **Hafner et al. (2020)** — "Dream to Control: Learning Behaviors by Latent Imagination"
   (DreamerV1). *Original imagination-based architecture.*
3. **Sutton (1991)** — "Dyna, an Integrated Architecture for Learning, Planning, and Reacting."
   *Foundational work on model-based RL with imagined transitions.*
4. **Mnih et al. (2015)** — "Human-level control through deep reinforcement learning" (DQN).
   *Context for model-free baselines.*

---

## 10. Resolved Decisions

> All tracked in `docs/decisions.md` with rationale.

- [x] **Imagination rollout length**: Set to H=5 (shorter than DreamerV3 due to simple MLP world model — see Decision 8)
- [x] **Network sizes**: 256×2 hidden layers for all MLPs
- [x] **Hyperparameters**: lr=3e-4, γ=0.99, batch=64 (standard defaults)
- [x] **Two-hot encoding**: Skipped — focus on symlog only (see Decision 3)
- [x] **Training approach**: Hybrid Dyna-style (real + imagined data) instead of pure imagination (see Decision 6)
- [x] **Numerical stability**: symexp clamping + gradient clipping (see Decision 7)

