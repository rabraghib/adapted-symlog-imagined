"""Abstract base class for all DRL agents.

Defines the minimal interface that both model-free and imagination-based
agents must implement, ensuring interchangeable usage in the training
loop.
"""

from abc import ABC, abstractmethod
from typing import Dict


class BaseAgent(ABC):
    """Base interface for reinforcement learning agents.

    Subclasses must implement :meth:`select_action` and
    :meth:`train_step`.  The :meth:`save` / :meth:`load` methods
    provide optional persistence hooks with default no-op behaviour.
    """

    @abstractmethod
    def select_action(self, state, deterministic: bool = False) -> int:
        """Choose an action given the current state.

        Args:
            state: Current environment observation (NumPy array or
                tensor).
            deterministic: If ``True`` use greedy action selection.

        Returns:
            Integer action index.
        """

    @abstractmethod
    def train_step(self, buffer) -> Dict[str, float]:
        """Perform one training update.

        Args:
            buffer: Replay buffer from which transitions are sampled.

        Returns:
            Dictionary of scalar loss metrics.
        """

    def save(self, path: str) -> None:
        """Save agent state to *path* (optional override)."""

    def load(self, path: str) -> None:
        """Load agent state from *path* (optional override)."""
