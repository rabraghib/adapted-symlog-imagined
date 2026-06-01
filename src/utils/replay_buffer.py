"""Vectorized experience replay buffer for reinforcement learning.

Pre-allocates contiguous NumPy arrays to eliminate Python deque overhead,
list comprehension bottlenecks, and slow list ziplining during sampling.
"""

from typing import Dict
import numpy as np
import torch
from torch import Tensor


class ReplayBuffer:
    """Fixed-capacity experience replay buffer with pre-allocated NumPy arrays.

    This implementation avoids Python collection overhead by using contiguous
    NumPy arrays. Tensors are generated using fast zero-copy sharing via
    `torch.from_numpy`.

    Args:
        capacity: Maximum number of transitions to store.
        device: Target torch device for sampled tensors (e.g., "cpu" or "cuda").
    """

    def __init__(self, capacity: int, device: str = "cpu") -> None:
        self.capacity = capacity
        self.device = device
        self.size = 0
        self.idx = 0
        self.initialized = False

    def _init_buffers(self, state_dim: int) -> None:
        """Lazily pre-allocate memory buffers based on observation dimension."""
        self.states = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.next_states = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.initialized = True

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
        if not self.initialized:
            self._init_buffers(state.shape[0])

        self.states[self.idx] = state
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_states[self.idx] = next_state
        self.dones[self.idx] = done

        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, Tensor]:
        """Sample a random mini-batch and convert to tensors.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Dictionary with keys ``states``, ``actions``, ``rewards``,
            ``next_states``, ``dones`` — all as :class:`Tensor` on
            ``self.device``.
        """
        idxs = np.random.randint(0, self.size, size=batch_size)

        return {
            "states": torch.from_numpy(self.states[idxs]).to(self.device),
            "actions": torch.from_numpy(self.actions[idxs]).to(self.device),
            "rewards": torch.from_numpy(self.rewards[idxs]).to(self.device),
            "next_states": torch.from_numpy(self.next_states[idxs]).to(self.device),
            "dones": torch.from_numpy(self.dones[idxs]).to(self.device),
        }

    def sample_states(self, batch_size: int) -> Tensor:
        """Sample random states only (e.g. for imagination start states).

        Args:
            batch_size: Number of states to sample.

        Returns:
            Tensor of shape ``(batch_size, state_dim)`` on
            ``self.device``.
        """
        idxs = np.random.randint(0, self.size, size=batch_size)
        return torch.from_numpy(self.states[idxs]).to(self.device)

    def __len__(self) -> int:
        """Return the current number of stored transitions."""
        return self.size
