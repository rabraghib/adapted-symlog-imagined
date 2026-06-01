"""Main training entry point.

Usage examples::

    # Model-free baseline
    python -m src.train --agent model_free --env CartPole-v1 --seed 0

    # Imagination agent without symlog
    python -m src.train --agent imagination --env CartPole-v1 --seed 0

    # Imagination agent with symlog
    python -m src.train --agent imagination --env CartPole-v1 --use-symlog --seed 0
"""

from __future__ import annotations

import argparse
import os
import random
import time
from collections import deque
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch

from src.agents.imagination_ac import ImaginationAC
from src.agents.model_free_ac import ModelFreeAC
from src.config import Config
from src.utils.logger import Logger
from src.utils.replay_buffer import ReplayBuffer


# ======================================================================
# Helpers
# ======================================================================


def set_seeds(seed: int) -> None:
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic CUDNN (may hurt perf but ensures reproducibility)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_env(env_name: str, seed: int) -> gym.Env:
    """Create and seed a Gymnasium environment."""
    env = gym.make(env_name)
    env.reset(seed=seed)
    return env


def get_env_dims(env: gym.Env) -> Tuple[int, int]:
    """Extract state and action dimensionalities from an environment."""
    state_dim = int(np.prod(env.observation_space.shape))
    assert isinstance(
        env.action_space, gym.spaces.Discrete
    ), "Only discrete action spaces are supported."
    action_dim = int(env.action_space.n)
    return state_dim, action_dim


def evaluate_agent(
    agent,
    env_name: str,
    num_episodes: int,
    seed: int,
) -> Tuple[float, float]:
    """Evaluate *agent* for *num_episodes* with deterministic actions.

    Args:
        agent: An agent implementing ``select_action``.
        env_name: Gymnasium environment identifier.
        num_episodes: Number of evaluation episodes.
        seed: Base seed; each episode uses ``seed + i``.

    Returns:
        Tuple of (mean_return, std_return).
    """
    returns = []
    for i in range(num_episodes):
        env = gym.make(env_name)
        state, _ = env.reset(seed=seed + i)
        episode_return = 0.0
        done = False
        while not done:
            action = agent.select_action(state, deterministic=True)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_return += float(reward)
            done = terminated or truncated
        env.close()
        returns.append(episode_return)

    returns_arr = np.array(returns)
    return float(returns_arr.mean()), float(returns_arr.std())


# ======================================================================
# Training loop
# ======================================================================


def train(config: Config) -> None:
    """Run the full training pipeline for a single experiment.

    Args:
        config: Fully-populated configuration dataclass.
    """
    # Reproducibility
    set_seeds(config.seed)

    # Experiment directory
    exp_name = config.get_experiment_name()
    exp_dir = os.path.join(config.experiment_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # Environment
    env = make_env(config.env_name, config.seed)
    state_dim, action_dim = get_env_dims(env)

    # Agent
    if config.agent_type == "model_free":
        agent = ModelFreeAC(state_dim, action_dim, config)
    elif config.agent_type == "imagination":
        agent = ImaginationAC(state_dim, action_dim, config)
    else:
        raise ValueError(f"Unknown agent_type: {config.agent_type!r}")

    # Replay buffer & logger
    device = config.get_device()
    buffer = ReplayBuffer(capacity=config.buffer_size, device=device)
    logger = Logger(log_dir=exp_dir, config=config)

    # Tracking
    episode_return = 0.0
    episode_length = 0
    episode_count = 0
    recent_returns: deque[float] = deque(maxlen=20)

    print(f"{'='*60}")
    print(f"  Experiment : {exp_name}")
    print(f"  Device     : {device}")
    print(f"  Agent      : {config.agent_type}")
    print(f"  Symlog     : {config.use_symlog}")
    print(f"  Total Steps: {config.total_steps:,}")
    print(f"{'='*60}\n")

    state, _ = env.reset(seed=config.seed)
    start_time = time.time()

    for step in range(1, config.total_steps + 1):
        # 1. Select action
        action = agent.select_action(state)

        # 2. Step environment
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # 3. Store transition
        buffer.add(state, action, float(reward), next_state, float(done))

        episode_return += float(reward)
        episode_length += 1

        # 4. Train agent
        metrics = agent.train_step(buffer)
        if metrics:
            logger.log_step(step, metrics)

        # 5. Episode boundary
        if done:
            episode_count += 1
            recent_returns.append(episode_return)

            # Log episode metrics
            ep_metrics = {
                "episode_return": episode_return,
                "episode_length": episode_length,
            }
            logger.log_step(step, ep_metrics)

            # Progress print every episode
            elapsed = time.time() - start_time
            avg_ret = np.mean(recent_returns)
            fps = step / max(elapsed, 1e-8)
            print(
                f"  Step {step:>7,}/{config.total_steps:,}  |  "
                f"Ep {episode_count:>4}  |  "
                f"Return {episode_return:>7.1f}  |  "
                f"Avg(20) {avg_ret:>7.1f}  |  "
                f"FPS {fps:>6.0f}"
            )

            # Reset
            episode_return = 0.0
            episode_length = 0
            state, _ = env.reset()
        else:
            state = next_state

        # 6. Periodic evaluation
        if step % config.eval_freq == 0:
            mean_ret, std_ret = evaluate_agent(
                agent, config.env_name, config.eval_episodes, config.seed + 1000
            )
            logger.log_eval(step, mean_ret, std_ret)
            print(
                f"  [EVAL] Step {step:>7,}  |  "
                f"Mean Return {mean_ret:>7.1f} ± {std_ret:>5.1f}"
            )

    # ------------------------------------------------------------------
    # Wrap up
    # ------------------------------------------------------------------
    env.close()
    logger.save()
    agent.save(os.path.join(exp_dir, "agent.pt"))

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  Training complete — {total_time:.1f}s")
    print(f"  Results saved to: {exp_dir}")
    print(f"{'='*60}\n")


# ======================================================================
# CLI
# ======================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a DRL agent with optional imagination and symlog.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="imagination",
        choices=["model_free", "imagination"],
        help="Agent type.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="CartPole-v1",
        help="Gymnasium environment name.",
    )
    parser.add_argument(
        "--use-symlog",
        action="store_true",
        default=False,
        help="Enable symlog loss adaptation (only affects imagination agent).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=100_000,
        help="Total number of environment steps.",
    )
    parser.add_argument(
        "--experiment-dir",
        type=str,
        default="experiments",
        help="Root directory for experiment outputs.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: parse CLI args, build Config, and launch training."""
    args = parse_args()

    config = Config(
        env_name=args.env,
        agent_type=args.agent,
        use_symlog=args.use_symlog,
        seed=args.seed,
        total_steps=args.total_steps,
        experiment_dir=args.experiment_dir,
    )

    train(config)


if __name__ == "__main__":
    main()
