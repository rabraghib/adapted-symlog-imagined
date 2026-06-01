"""Discrete-action policy network (Actor).

Outputs a categorical distribution over actions given a state observation.
"""

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Categorical

from src.models.networks import build_mlp


class Actor(nn.Module):
    """Policy network for discrete action spaces.

    The network maps state observations to logits over actions and
    wraps them in a :class:`torch.distributions.Categorical` for
    convenient sampling and log-probability computation.

    Args:
        state_dim: Dimensionality of the state observation.
        action_dim: Number of discrete actions.
        hidden_dims: Widths of the hidden layers.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.network = build_mlp(state_dim, action_dim, hidden_dims)

    def forward(self, state: Tensor) -> Categorical:
        """Return a categorical distribution over actions.

        Args:
            state: Observation tensor, shape ``(*, state_dim)``.

        Returns:
            :class:`Categorical` distribution parameterised by the
            network logits.
        """
        logits = self.network(state)
        return Categorical(logits=logits)

    def get_action(
        self, state: Tensor, deterministic: bool = False
    ) -> Tuple[Tensor, Tensor]:
        """Select an action and compute its log-probability.

        Args:
            state: Observation tensor (may be batched or unbatched).
            deterministic: If ``True``, select the action with the
                highest probability (greedy).

        Returns:
            action: Selected action index (``int``-like tensor).
            log_prob: Log-probability of the selected action.
        """
        dist = self.forward(state)
        if deterministic:
            action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def evaluate(
        self, state: Tensor, action: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Evaluate a given action under the current policy.

        Useful for computing the policy-gradient loss on a batch of
        previously collected transitions.

        Args:
            state: Observation tensor, shape ``(B, state_dim)``.
            action: Action indices, shape ``(B,)``.

        Returns:
            log_prob: Log-probability of each action, shape ``(B,)``.
            entropy: Per-sample entropy of the distribution,
                shape ``(B,)``.
        """
        dist = self.forward(state)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy
