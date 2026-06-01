"""Symlog loss functions for scale-invariant value and reward prediction.

The symmetric logarithmic (symlog) transform compresses large-magnitude
targets while preserving their sign, improving gradient stability for
reward and value prediction in reinforcement learning.

Reference:
    Hafner et al., "Mastering Diverse Domains through World Models"
    (DreamerV3), 2023.
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def symlog(x: Tensor) -> Tensor:
    """Symmetric logarithmic transformation: ``sign(x) * ln(|x| + 1)``.

    Args:
        x: Input tensor of arbitrary shape.

    Returns:
        Transformed tensor with the same shape as *x*.
    """
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: Tensor) -> Tensor:
    """Inverse of :func:`symlog`: ``sign(x) * (exp(|x|) - 1)``.

    The input is clamped to ``[-20, 20]`` to prevent numerical overflow
    in the exponential (``exp(20) ≈ 4.8e8`` which is safe for float32).

    Args:
        x: Input tensor in symlog space.

    Returns:
        Tensor mapped back to the original (linear) scale.
    """
    return torch.sign(x) * (torch.exp(torch.clamp(torch.abs(x), max=20.0)) - 1)


def symlog_mse_loss(predictions: Tensor, targets: Tensor) -> Tensor:
    """MSE loss computed in symlog space.

    Targets are transformed via :func:`symlog` before comparison.
    **Predictions are assumed to already be in symlog space** (i.e. the
    network's raw output).

    Args:
        predictions: Predicted values in symlog space.
        targets: Ground-truth values in *original* (linear) scale.

    Returns:
        Scalar MSE loss.
    """
    return F.mse_loss(predictions, symlog(targets))


def standard_mse_loss(predictions: Tensor, targets: Tensor) -> Tensor:
    """Standard MSE loss (provided for API consistency).

    Args:
        predictions: Predicted values.
        targets: Ground-truth values.

    Returns:
        Scalar MSE loss.
    """
    return F.mse_loss(predictions, targets)
