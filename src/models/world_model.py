"""World model with dynamics and reward prediction.

The world model learns to predict environment transitions (next state)
and rewards given a current state and action.  It supports an optional
symlog output space for reward prediction, which improves training
stability across environments with varying reward scales.
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.losses.symlog import symexp, symlog_mse_loss
from src.models.networks import build_mlp


class WorldModel(nn.Module):
    """Learned world model for imagination-based planning.

    The model consists of two heads:

    * **Dynamics model** – predicts a *residual* (delta) added to the
      current state to obtain the next state, which improves learning
      stability.
    * **Reward model** – predicts the scalar reward.  When
      ``use_symlog=True`` the raw output lives in symlog space.

    Args:
        state_dim: Dimensionality of the observation / state vector.
        action_dim: Number of discrete actions (one-hot width).
        hidden_dims: Widths of hidden layers for both heads.
        use_symlog: If ``True``, reward predictions are made in symlog
            space and converted back via :func:`symexp` when needed.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        use_symlog: bool = False,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_symlog = use_symlog

        input_dim = state_dim + action_dim
        self.dynamics_model = build_mlp(input_dim, state_dim, hidden_dims)
        self.reward_model = build_mlp(input_dim, 1, hidden_dims)

    def forward(
        self, state: Tensor, action_onehot: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Predict the next state and raw reward.

        Args:
            state: Current states, shape ``(B, state_dim)``.
            action_onehot: One-hot encoded actions, shape ``(B, action_dim)``.

        Returns:
            predicted_next_state: ``state + delta`` from the dynamics
                model, shape ``(B, state_dim)``.
            predicted_reward_raw: Raw reward output (in symlog space when
                ``use_symlog`` is enabled), shape ``(B,)``.
        """
        concat = torch.cat([state, action_onehot], dim=-1)
        delta = self.dynamics_model(concat)
        predicted_next_state = state + delta  # residual prediction
        predicted_reward_raw = self.reward_model(concat).squeeze(-1)
        return predicted_next_state, predicted_reward_raw

    def predict_reward(self, state: Tensor, action_onehot: Tensor) -> Tensor:
        """Return the reward prediction in the *original* (linear) scale.

        Args:
            state: Current states, shape ``(B, state_dim)``.
            action_onehot: One-hot encoded actions, shape ``(B, action_dim)``.

        Returns:
            Predicted rewards, shape ``(B,)``.
        """
        _, raw_reward = self.forward(state, action_onehot)
        if self.use_symlog:
            return symexp(raw_reward)
        return raw_reward

    def train_step(
        self,
        states: Tensor,
        actions_onehot: Tensor,
        rewards: Tensor,
        next_states: Tensor,
        optimizer: torch.optim.Optimizer,
        grad_clip_norm: float = 1.0,
    ) -> Dict[str, float]:
        """Perform a single supervised training step.

        Args:
            states: Batch of current states, shape ``(B, state_dim)``.
            actions_onehot: One-hot actions, shape ``(B, action_dim)``.
            rewards: Ground-truth rewards, shape ``(B,)``.
            next_states: Ground-truth next states, shape ``(B, state_dim)``.
            optimizer: Optimizer for this world model's parameters.
            grad_clip_norm: Maximum gradient norm for clipping.

        Returns:
            Dictionary with scalar loss values:
            ``dynamics_loss``, ``reward_loss``, ``total_wm_loss``.
        """
        predicted_next_state, predicted_reward_raw = self.forward(
            states, actions_onehot
        )

        dynamics_loss = F.mse_loss(predicted_next_state, next_states)

        if self.use_symlog:
            reward_loss = symlog_mse_loss(predicted_reward_raw, rewards)
        else:
            reward_loss = F.mse_loss(predicted_reward_raw, rewards)

        total_loss = dynamics_loss + reward_loss

        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), grad_clip_norm)
        optimizer.step()

        return {
            "dynamics_loss": dynamics_loss.item(),
            "reward_loss": reward_loss.item(),
            "total_wm_loss": total_loss.item(),
        }

    def imagine(
        self,
        start_states: Tensor,
        actor: nn.Module,
        horizon: int,
    ) -> Dict[str, object]:
        """Unroll imagined trajectories through the learned world model.

        During imagination the world model's parameters are **not**
        updated (predictions are made under ``torch.no_grad()`` for the
        world model), while the actor's computational graph is preserved
        so that policy gradients can flow through the imagined
        log-probabilities.

        Args:
            start_states: Initial states for imagination, shape
                ``(B, state_dim)``.
            actor: Policy network whose ``forward`` returns a
                :class:`torch.distributions.Categorical`.
            horizon: Number of imagination steps to unroll.

        Returns:
            Dictionary containing:
            - ``states``: list of *T* tensors, each ``(B, state_dim)``.
            - ``actions``: list of *T* tensors, each ``(B,)``.
            - ``rewards``: list of *T* tensors, each ``(B,)`` in
              original scale.
            - ``log_probs``: list of *T* tensors, each ``(B,)`` (with
              gradients for the actor).
            - ``final_states``: last predicted states ``(B, state_dim)``
              for value bootstrapping.
        """
        states_list: List[Tensor] = []
        actions_list: List[Tensor] = []
        rewards_list: List[Tensor] = []
        log_probs_list: List[Tensor] = []

        current_states = start_states

        for _ in range(horizon):
            # Actor forward — keep gradients for policy optimisation.
            dist = actor(current_states)
            actions = dist.sample()
            log_probs_h = dist.log_prob(actions)

            action_onehot = F.one_hot(
                actions, num_classes=actor.action_dim
            ).float()

            # World-model forward — no gradient through the model weights.
            with torch.no_grad():
                next_states, reward_raw = self.forward(
                    current_states, action_onehot
                )

            # Convert reward to original scale.
            if self.use_symlog:
                reward = symexp(reward_raw)
            else:
                reward = reward_raw

            states_list.append(current_states)
            actions_list.append(actions)
            rewards_list.append(reward)
            log_probs_list.append(log_probs_h)

            # Detach to prevent back-prop through dynamics for the actor.
            current_states = next_states.detach()

        return {
            "states": states_list,
            "actions": actions_list,
            "rewards": rewards_list,
            "log_probs": log_probs_list,
            "final_states": current_states,
        }
