"""Configuration dataclass for the DRL project.

Provides a centralized, dataclass-based configuration for all hyperparameters,
environment settings, network architecture, and experiment metadata.
"""

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class Config:
    """Central configuration for the Dreamer-like imagination-based DRL agent.

    Attributes:
        env_name: Gymnasium environment identifier.
        agent_type: Either ``"model_free"`` or ``"imagination"``.
        use_symlog: Whether to apply symlog loss adaptation for reward/value
            prediction.
        total_steps: Total number of environment interaction steps.
        batch_size: Mini-batch size for network updates.
        buffer_size: Maximum capacity of the replay buffer.
        learning_rate: Shared learning rate for all optimizers.
        gamma: Discount factor for return computation.
        imagination_horizon: Number of steps to unroll during imagination.
        num_imagine_batches: Number of batches sampled for imagination rollouts.
        world_model_train_freq: Train the world model every *N* environment
            steps.
        hidden_dims: Tuple of hidden-layer widths shared across all MLPs.
        entropy_coef: Entropy bonus coefficient for the actor loss.
        eval_freq: Evaluate the agent every *N* environment steps.
        eval_episodes: Number of episodes per evaluation.
        seed: Random seed for reproducibility.
        device: Torch device string (``"auto"``, ``"cuda"``, or ``"cpu"``).
        experiment_dir: Root directory for experiment outputs.
    """

    # Environment
    env_name: str = "CartPole-v1"

    # Agent
    agent_type: str = "imagination"  # "model_free" or "imagination"
    use_symlog: bool = False

    # Training
    total_steps: int = 100_000
    batch_size: int = 64
    buffer_size: int = 100_000
    learning_rate: float = 3e-4
    gamma: float = 0.99

    # Imagination
    imagination_horizon: int = 5
    num_imagine_batches: int = 8
    world_model_train_freq: int = 1
    warmup_steps: int = 1000  # Collect data before starting imagination
    world_model_train_steps: int = 2  # WM gradient steps per train call
    imagination_train_ratio: int = 10  # Do imagination update every N train calls
    grad_clip_norm: float = 1.0  # Max gradient norm for stability

    # Networks
    hidden_dims: Tuple[int, ...] = (256, 256)

    # Actor
    entropy_coef: float = 0.01

    # Logging
    eval_freq: int = 1_000
    eval_episodes: int = 10
    seed: int = 0

    # Device
    device: str = "auto"  # "auto", "cuda", "cpu"

    # Experiment output
    experiment_dir: str = "experiments"

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def get_device(self) -> torch.device:
        """Resolve the ``device`` field to a concrete :class:`torch.device`.

        ``"auto"`` maps to CUDA when available, otherwise CPU.
        """
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def get_experiment_name(self) -> str:
        """Return a human-readable experiment identifier.

        Format: ``{env_name}_{agent_type}[_symlog]_seed{seed}``
        """
        parts = [self.env_name, self.agent_type]
        if self.use_symlog:
            parts.append("symlog")
        parts.append(f"seed{self.seed}")
        return "_".join(parts)
