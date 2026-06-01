# Technical Decisions Log

Track all design decisions here with rationale, so they're easy to reference in the report.

---

## Decision 1: Actor-Critic vs DQN as base algorithm
- **Decision**: Use Actor-Critic (policy gradient + value function), not DQN
- **Rationale**: The assignment says "imagination-based model," which refers to
  the Dreamer family. Dreamer uses actor-critic trained on imagined trajectories.
  DQN + world model would be Dyna-Q style — valid but not what the literature
  means by "imagination-based." Actor-Critic also makes the symlog integration
  cleaner (applied to critic value targets).
- **Date**: 2026-06-01

## Decision 2: Work in raw state space, not latent space
- **Decision**: Our world model operates on raw environment states (vectors),
  not learned latent representations
- **Rationale**: Full DreamerV3 uses an RSSM (Recurrent State-Space Model) to
  learn compact latent states — this is complex to implement correctly in 2 weeks.
  CartPole and LunarLander have small, meaningful state vectors (4D and 8D), so
  there's no need for latent encoding. We mention this simplification in the report.
- **Date**: 2026-06-01

## Decision 3: Skip two-hot encoding
- **Decision**: Use direct symlog MSE loss, not two-hot discrete regression
- **Rationale**: DreamerV3 combines symlog with two-hot encoding (discretizing
  targets into 255 buckets + cross-entropy loss). This is an orthogonal technique
  that adds implementation complexity without being core to the symlog story.
  We mention it in the report as part of the full DreamerV3 approach.
- **Date**: 2026-06-01

## Decision 4: Discrete action spaces only
- **Decision**: Use CartPole-v1 and LunarLander-v3 (both discrete action spaces)
- **Rationale**: Continuous action spaces require additional complexity (e.g.,
  reparameterization trick, squashed Gaussian policies). Discrete actions keep
  the actor simple (categorical policy) so the focus stays on the loss function
  adaptation, which is the actual subject of the project.
- **Date**: 2026-06-01

## Decision 5: Three experimental conditions
- **Decision**: Compare Model-Free AC vs Imagination AC (MSE) vs Imagination AC (Symlog)
- **Rationale**: Two comparisons are needed to tell the full story:
  - A→B shows the benefit of imagination-based training
  - B→C isolates the effect of symlog specifically
  If we only compared A vs C, we couldn't disentangle world model benefits from symlog benefits.
- **Date**: 2026-06-01

## Decision 6: Hybrid (Dyna-style) training, not pure imagination
- **Decision**: Train actor-critic on BOTH real transitions AND imagined rollouts,
  rather than purely on imagined rollouts like full DreamerV3
- **Rationale**: During testing, pure imagination-based training collapsed to
  minimum performance (~9 steps on CartPole) because the world model's inaccuracies
  propagated through imagined trajectories and destroyed the policy. The hybrid
  approach (Sutton's Dyna, 1991) keeps the policy grounded in real data while
  still benefiting from imagined experience. This is a well-known approach in
  model-based RL and is appropriate for our simplified (non-latent) world model.
- **Implementation**: Real-data AC updates happen every step; imagination updates
  happen every `imagination_train_ratio` steps (default 10).
- **Date**: 2026-06-01

## Decision 7: Numerical stability for symlog
- **Decision**: Clamp symexp input to [-20, 20], use gradient clipping (norm=1.0)
- **Rationale**: During testing, the symlog variant crashed with NaN values.
  `symexp(x)` calls `exp(|x|)` which overflows float32 for |x| > ~88. Clamping
  to 20 (exp(20) ≈ 4.8e8) prevents overflow while allowing a huge dynamic range.
  Gradient clipping prevents the world model and actor-critic from producing
  extreme parameter updates early in training.
- **Date**: 2026-06-01

## Decision 8: Short imagination horizon (5 steps)
- **Decision**: Use horizon=5 instead of DreamerV3's longer horizons
- **Rationale**: Our world model is a simple MLP operating in raw state space,
  unlike DreamerV3's RSSM which learns stable latent dynamics. Compounding
  prediction errors grow exponentially with horizon length. Shorter horizons
  keep imagined trajectories closer to reality.
- **Date**: 2026-06-01

## Decision 9: Subprocess-level parallel execution with thread limiting
- **Decision**: Limit PyTorch and BLAS/LAPACK threads to 1 per subprocess when running parallel experiments
- **Rationale**: Spawning 30 parallel subprocesses where each attempts to use all available CPU cores causes massive CPU thread thrashing (over-subscription), dropping performance (FPS down to ~1). Constraining each subprocess to a single thread (`torch.set_num_threads(1)` and thread-limiting environment variables like `OMP_NUM_THREADS=1`) allows clean core scaling, reducing total runtime for the 30-experiment suite to just ~100 minutes on a 32-core machine.
- **Date**: 2026-06-01

## Decision 10: Pre-allocated NumPy array Replay Buffer (Vectorized Buffer)
- **Decision**: Replace Python `deque` of tuples with a pre-allocated contiguous NumPy array-backed replay buffer
- **Rationale**: Python `deque` of tuples is slow for sampling large batch sizes due to list copying and Python object allocations. Pre-allocating contiguous NumPy arrays once at start allows vectorized indexing with `np.random.randint()` and zero-copy tensor sharing via `torch.from_numpy()`. This significantly reduces training overhead, which is critical when multiple runs execute concurrently and compete for memory bandwidth.
- **Date**: 2026-06-01
