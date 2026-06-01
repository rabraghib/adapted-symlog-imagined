"""Shared MLP builder used by all neural-network modules.

Provides a single factory function that constructs feed-forward networks
with configurable depth, width, and activation function.
"""

from typing import Tuple, Type

import torch.nn as nn


def build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: Tuple[int, ...] = (256, 256),
    activation: Type[nn.Module] = nn.ReLU,
) -> nn.Sequential:
    """Build a multi-layer perceptron.

    Args:
        input_dim: Dimensionality of the input features.
        output_dim: Dimensionality of the output layer.
        hidden_dims: Widths of each hidden layer.
        activation: Activation class inserted after each hidden layer.

    Returns:
        An :class:`nn.Sequential` module representing the MLP.
    """
    layers: list[nn.Module] = []
    prev_dim = input_dim
    for h_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, h_dim))
        layers.append(activation())
        prev_dim = h_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)
