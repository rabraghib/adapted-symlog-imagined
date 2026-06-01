"""Model-free Actor-Critic agent (Condition A).

A standard advantage actor-critic that trains exclusively on real environment
transitions.  This serves as the baseline condition in the experiment — no
world model or imagined rollouts are involved.
"""

from __future__ import annotations

import os
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from src.agents.base import BaseAgent
from src.config import Config
from src.models.actor import Actor
from src.models.critic import Critic
from src.utils.replay_buffer import ReplayBuffer


class ModelFreeAC(BaseAgent):
    """Model-free advantage actor-critic.

    Uses real transitions sampled from a replay buffer to compute single-step
    (TD(0)) target returns and updates the actor with the policy-gradient
    theorem plus an entropy bonus.

    Args:
        state_dim: Dimensionality of the observation space.
        action_dim: Number of discrete actions.
        config: Experiment configuration dataclass.
    """

    def __init__(self, state_dim: int, action_dim: int, config: Config) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.device = config.get_device()

        # Networks — model-free always uses standard (non-symlog) critic
        self.actor = Actor(state_dim, action_dim, hidden_dims=config.hidden_dims).to(
            self.device
        )
        self.critic = Critic(
            state_dim, hidden_dims=config.hidden_dims, use_symlog=False
        ).to(self.device)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=config.learning_rate
        )

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        """Select an action given an observation.

        Args:
            state: NumPy array of shape ``(state_dim,)``.
            deterministic: If *True*, pick the mode of the policy.

        Returns:
            Scalar integer action.
        """
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.actor.get_action(state_t, deterministic=deterministic)
        return int(action)

    def train_step(self, buffer: ReplayBuffer) -> Dict[str, float]:
        """Perform one training update from a batch of real transitions.

        Returns an empty dict when the buffer has fewer samples than the
        configured batch size.

        Args:
            buffer: Replay buffer to sample from.

        Returns:
            Dictionary of scalar loss metrics.
        """
        if len(buffer) < self.config.batch_size:
            return {}

        batch = buffer.sample(self.config.batch_size)
        states = batch["states"]       # (B, state_dim)
        actions = batch["actions"]     # (B,) long
        rewards = batch["rewards"]     # (B,)
        next_states = batch["next_states"]  # (B, state_dim)
        dones = batch["dones"]         # (B,)

        # ----------------------------------------------------------
        # 1. Compute TD(0) target returns
        # ----------------------------------------------------------
        with torch.no_grad():
            next_values = self.critic.get_value(next_states)  # (B,)
            targets = rewards + self.config.gamma * next_values * (1.0 - dones)  # (B,)

        # ----------------------------------------------------------
        # 2. Critic update
        # ----------------------------------------------------------
        critic_loss = self.critic.compute_loss(states, targets)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ----------------------------------------------------------
        # 3. Actor update
        # ----------------------------------------------------------
        with torch.no_grad():
            current_values = self.critic.get_value(states)  # (B,)
            advantages = targets - current_values             # (B,)

        log_probs, entropy = self.actor.evaluate(states, actions)

        actor_loss = -(log_probs * advantages.detach()).mean() - (
            self.config.entropy_coef * entropy.mean()
        )

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "entropy": entropy.mean().item(),
            "mean_advantage": advantages.mean().item(),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save all model and optimizer state dicts to *path*."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load all model and optimizer state dicts from *path*."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
