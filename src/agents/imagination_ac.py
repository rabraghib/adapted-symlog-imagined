"""Imagination-based Actor-Critic agent (Conditions B & C).

A simplified Dreamer-like agent that uses a **hybrid training** approach:
1. Trains a learned world model on real transitions.
2. Trains the actor and critic on **real transitions** (ensures policy
   stays grounded in actual environment dynamics).
3. Additionally trains the actor and critic on **imagined rollouts** from
   the world model (provides extra learning signal).

This hybrid (Dyna-style) approach avoids the failure mode where a purely
imagination-trained policy diverges due to world model inaccuracies.

When ``use_symlog`` is enabled (Condition C), the critic and world model
reward heads operate in symlog-transformed space, applying ``symexp`` when
values in the original scale are required.
"""

from __future__ import annotations

import os
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.agents.base import BaseAgent
from src.config import Config
from src.models.actor import Actor
from src.models.critic import Critic
from src.models.world_model import WorldModel
from src.utils.replay_buffer import ReplayBuffer


class ImaginationAC(BaseAgent):
    """Imagination-based advantage actor-critic with optional symlog.

    Uses a Dyna-style hybrid approach: the actor and critic are trained
    on both real transitions and imagined rollouts from the learned
    world model.

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

        # Networks
        self.actor = Actor(
            state_dim, action_dim, hidden_dims=config.hidden_dims
        ).to(self.device)

        self.critic = Critic(
            state_dim, hidden_dims=config.hidden_dims, use_symlog=config.use_symlog
        ).to(self.device)

        self.world_model = WorldModel(
            state_dim,
            action_dim,
            hidden_dims=config.hidden_dims,
            use_symlog=config.use_symlog,
        ).to(self.device)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=config.learning_rate
        )
        self.world_model_optimizer = torch.optim.Adam(
            self.world_model.parameters(), lr=config.learning_rate
        )

        # Book-keeping
        self.train_counter: int = 0
        self.total_env_steps: int = 0  # Track total steps for warmup

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
        self.total_env_steps += 1

        # During warmup, use random actions for diverse exploration
        if self.total_env_steps <= self.config.warmup_steps and not deterministic:
            return int(np.random.randint(self.action_dim))

        state_t = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.actor.get_action(state_t, deterministic=deterministic)
        return int(action)

    def _update_actor_critic_on_real(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Train actor and critic on a batch of real transitions (TD(0)).

        This mirrors the model-free AC update and keeps the policy
        grounded in real environment dynamics.

        Args:
            batch: Dictionary from ReplayBuffer.sample().

        Returns:
            Dictionary of loss metrics.
        """
        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_states"]
        dones = batch["dones"]

        # Compute TD(0) targets
        with torch.no_grad():
            next_values = self.critic.get_value(next_states)
            targets = rewards + self.config.gamma * next_values * (1.0 - dones)

        # Critic update
        critic_loss = self.critic.compute_loss(states, targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.grad_clip_norm)
        self.critic_optimizer.step()

        # Actor update
        with torch.no_grad():
            current_values = self.critic.get_value(states)
            advantages = targets - current_values

        log_probs, entropy = self.actor.evaluate(states, actions)
        actor_loss = -(log_probs * advantages.detach()).mean() - (
            self.config.entropy_coef * entropy.mean()
        )
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.grad_clip_norm)
        self.actor_optimizer.step()

        return {
            "real_critic_loss": critic_loss.item(),
            "real_actor_loss": actor_loss.item(),
        }

    def _update_actor_critic_on_imagined(self, buffer: ReplayBuffer) -> Dict[str, float]:
        """Train actor and critic on imagined rollouts from the world model.

        This provides additional learning signal beyond real transitions,
        allowing the agent to learn from simulated future trajectories.

        Args:
            buffer: Replay buffer to sample start states from.

        Returns:
            Dictionary of loss metrics.
        """
        # Sample start states for imagination
        num_start = self.config.batch_size
        start_states = buffer.sample_states(num_start).detach()

        # Imagine forward
        imagined = self.world_model.imagine(
            start_states, self.actor, self.config.imagination_horizon
        )

        horizon = self.config.imagination_horizon

        # Compute imagined returns (backwards, no grad)
        with torch.no_grad():
            final_value = self.critic.get_value(imagined["final_states"])

            returns = []
            for _ in range(horizon):
                returns.append(torch.zeros_like(final_value))

            returns[-1] = imagined["rewards"][-1] + self.config.gamma * final_value
            for h in range(horizon - 2, -1, -1):
                returns[h] = (
                    imagined["rewards"][h] + self.config.gamma * returns[h + 1]
                )

        # Stack across horizon: (H*N, ...)
        all_states = torch.cat(imagined["states"], dim=0)
        all_returns = torch.cat(returns, dim=0)
        all_actions = torch.cat(imagined["actions"], dim=0)

        # Critic update on imagined data
        critic_loss = self.critic.compute_loss(all_states, all_returns)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.grad_clip_norm)
        self.critic_optimizer.step()

        # Actor update on imagined data
        log_probs, entropy = self.actor.evaluate(all_states.detach(), all_actions)

        with torch.no_grad():
            advantages = all_returns - self.critic.get_value(all_states)
            if advantages.numel() > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_loss = -(log_probs * advantages).mean() - (
            self.config.entropy_coef * entropy.mean()
        )
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.grad_clip_norm)
        self.actor_optimizer.step()

        return {
            "imag_critic_loss": critic_loss.item(),
            "imag_actor_loss": actor_loss.item(),
            "entropy": entropy.mean().item(),
            "mean_imagined_reward": torch.cat(imagined["rewards"]).mean().item(),
        }

    def train_step(self, buffer: ReplayBuffer) -> Dict[str, float]:
        """Perform one full training iteration (hybrid: real + imagined).

        The training has two phases:

        **Phase 1 (always)**: Train the world model on real transitions,
        then train the actor-critic on real transitions (identical to
        model-free AC, keeps the policy grounded).

        **Phase 2 (after warmup)**: Additionally train the actor-critic
        on imagined rollouts from the world model (extra learning signal).

        Args:
            buffer: Replay buffer containing real transitions.

        Returns:
            Dictionary of scalar loss metrics.
        """
        if len(buffer) < self.config.batch_size:
            return {}

        self.train_counter += 1
        metrics: Dict[str, float] = {}

        # ==============================================================
        # Phase 1A: Train world model on real data (multiple steps)
        # ==============================================================
        for _ in range(self.config.world_model_train_steps):
            batch = buffer.sample(self.config.batch_size)
            actions_onehot = F.one_hot(
                batch["actions"].long(), num_classes=self.action_dim
            ).float()

            wm_metrics = self.world_model.train_step(
                batch["states"], actions_onehot, batch["rewards"],
                batch["next_states"], self.world_model_optimizer,
                grad_clip_norm=self.config.grad_clip_norm,
            )
        metrics.update(wm_metrics)

        # ==============================================================
        # Phase 1B: Train actor-critic on REAL transitions (always)
        # ==============================================================
        real_batch = buffer.sample(self.config.batch_size)
        real_metrics = self._update_actor_critic_on_real(real_batch)
        metrics.update(real_metrics)

        # ==============================================================
        # Phase 2: Train actor-critic on IMAGINED trajectories
        # (after warmup, every N train steps)
        # ==============================================================
        past_warmup = self.total_env_steps > self.config.warmup_steps
        imagination_step = (self.train_counter % self.config.imagination_train_ratio == 0)
        if past_warmup and imagination_step:
            imag_metrics = self._update_actor_critic_on_imagined(buffer)
            metrics.update(imag_metrics)

        return metrics

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save all model and optimizer state dicts to *path*."""
        os.makedirs(
            os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True
        )
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "world_model": self.world_model.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "world_model_optimizer": self.world_model_optimizer.state_dict(),
                "train_counter": self.train_counter,
                "total_env_steps": self.total_env_steps,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load all model and optimizer state dicts from *path*."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.world_model.load_state_dict(checkpoint["world_model"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.world_model_optimizer.load_state_dict(
            checkpoint["world_model_optimizer"]
        )
        self.train_counter = checkpoint.get("train_counter", 0)
        self.total_env_steps = checkpoint.get("total_env_steps", 0)
