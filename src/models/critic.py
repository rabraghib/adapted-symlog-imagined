"""State-value network (Critic) with optional symlog awareness.

When ``use_symlog=True`` the network's raw output is treated as a value
in symlog space, and the training target is transformed via
:func:`~src.losses.symlog.symlog` before computing MSE.  The
:meth:`get_value` method always returns predictions in the *original*
(linear) scale.
"""

from typing import Tuple

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.losses.symlog import symexp, symlog
from src.models.networks import build_mlp


class Critic(nn.Module):
    """Value network for state-value estimation.

    Args:
        state_dim: Dimensionality of the state observation.
        hidden_dims: Widths of the hidden layers.
        use_symlog: If ``True``, predictions and targets live in symlog
            space during training; :meth:`get_value` converts back.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        use_symlog: bool = False,
    ) -> None:
        super().__init__()
        self.network = build_mlp(state_dim, 1, hidden_dims)
        self.use_symlog = use_symlog

    def forward(self, state: Tensor) -> Tensor:
        """Return the *raw* network output (one scalar per state).

        In symlog mode this value is in symlog space.

        Args:
            state: Observation tensor, shape ``(B, state_dim)``.

        Returns:
            Scalar value predictions, shape ``(B,)``.
        """
        return self.network(state).squeeze(-1)

    def get_value(self, state: Tensor) -> Tensor:
        """Return value estimates in the *original* (linear) scale.

        Args:
            state: Observation tensor, shape ``(B, state_dim)``.

        Returns:
            Value predictions, shape ``(B,)``.
        """
        raw = self.forward(state)
        if self.use_symlog:
            return symexp(raw)
        return raw

    def compute_loss(self, states: Tensor, target_returns: Tensor) -> Tensor:
        """Compute the value-prediction loss.

        When ``use_symlog`` is enabled, the target returns are mapped
        into symlog space and compared against the raw network output.

        Args:
            states: Observation tensor, shape ``(B, state_dim)``.
            target_returns: Ground-truth returns, shape ``(B,)``, in
                the *original* (linear) scale.

        Returns:
            Scalar MSE loss.
        """
        raw_predictions = self.forward(states)
        if self.use_symlog:
            return F.mse_loss(raw_predictions, symlog(target_returns))
        return F.mse_loss(raw_predictions, target_returns)
