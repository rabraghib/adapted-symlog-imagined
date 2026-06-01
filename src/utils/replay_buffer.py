"""Simple replay buffer for off-policy reinforcement learning.

Stores transitions as NumPy arrays and lazily converts sampled batches
to PyTorch tensors on the requested device.
"""

import random
from collections import deque
from typing import Dict

import numpy as np
import torch
from torch import Tensor


class ReplayBuffer:
    """Fixed-capacity experience replay buffer.

    Transitions are stored as tuples of NumPy arrays.  Sampling
    converts them on-the-fly to :class:`torch.Tensor` placed on the
    specified device.

    Args:
        capacity: Maximum number of transitions to store.
        device: Target torch device for sampled tensors (e.g.
            ``"cpu"`` or ``"cuda"``).
    """

    def __init__(self, capacity: int, device: str = "cpu") -> None:
        self.buffer: deque = deque(maxlen=capacity)
        self.device = device

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Append a single transition to the buffer.

        Args:
            state: Current observation (1-D NumPy array).
            action: Selected action index.
            reward: Scalar reward received.
            next_state: Successor observation (1-D NumPy array).
            done: Whether the episode terminated after this step.
        """
        self.buffer.append(
            (
                np.asarray(state, dtype=np.float32),
                int(action),
                float(reward),
                np.asarray(next_state, dtype=np.float32),
                float(done),
            )
        )

    def sample(self, batch_size: int) -> Dict[str, Tensor]:
        """Sample a random mini-batch and convert to tensors.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Dictionary with keys ``states``, ``actions``, ``rewards``,
            ``next_states``, ``dones`` — all as :class:`Tensor` on
            ``self.device``.
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return {
            "states": torch.tensor(
                np.array(states), dtype=torch.float32, device=self.device
            ),
            "actions": torch.tensor(
                actions, dtype=torch.long, device=self.device
            ),
            "rewards": torch.tensor(
                rewards, dtype=torch.float32, device=self.device
            ),
            "next_states": torch.tensor(
                np.array(next_states), dtype=torch.float32, device=self.device
            ),
            "dones": torch.tensor(
                dones, dtype=torch.float32, device=self.device
            ),
        }

    def sample_states(self, batch_size: int) -> Tensor:
        """Sample random states only (e.g. for imagination start states).

        Args:
            batch_size: Number of states to sample.

        Returns:
            Tensor of shape ``(batch_size, state_dim)`` on
            ``self.device``.
        """
        batch = random.sample(self.buffer, batch_size)
        states = np.array([t[0] for t in batch])
        return torch.tensor(states, dtype=torch.float32, device=self.device)

    def __len__(self) -> int:
        """Return the current number of stored transitions."""
        return len(self.buffer)
