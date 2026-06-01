"""Visualization helpers for training analysis.

All functions accept a ``results`` dictionary keyed by condition name
and save publication-quality PNG figures at 300 DPI.
"""

from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Use a non-interactive backend so plotting works in headless environments.
matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Style configuration
# ---------------------------------------------------------------------------
_PALETTE = [
    "#4C72B0",  # steel blue
    "#DD8452",  # sandy orange
    "#55A868",  # sage green
    "#C44E52",  # brick red
    "#8172B3",  # muted purple
    "#937860",  # warm brown
    "#DA8BC3",  # soft pink
    "#8C8C8C",  # neutral grey
    "#CCB974",  # gold
    "#64B5CD",  # sky blue
]


def _apply_style() -> None:
    """Apply a clean, modern matplotlib style."""
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        # Fallback for older matplotlib versions.
        try:
            plt.style.use("seaborn-whitegrid")
        except OSError:
            plt.style.use("ggplot")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plot_learning_curves(
    results: Dict[str, Dict[str, List[float]]],
    save_path: str,
    title: str = "Learning Curves",
) -> None:
    """Plot evaluation return curves with shaded standard-deviation bands.

    Args:
        results: Mapping from condition name to a dict with keys
            ``steps`` (list of ints), ``returns_mean`` (list of floats),
            and ``returns_std`` (list of floats).
        save_path: File path for the output PNG.
        title: Plot title.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    for idx, (name, data) in enumerate(results.items()):
        color = _PALETTE[idx % len(_PALETTE)]
        steps = np.asarray(data["steps"])
        mean = np.asarray(data["returns_mean"])
        std = np.asarray(data["returns_std"])

        ax.plot(steps, mean, label=name, color=color, linewidth=2)
        ax.fill_between(steps, mean - std, mean + std, alpha=0.2, color=color)

    ax.set_xlabel("Environment Steps", fontsize=13)
    ax.set_ylabel("Evaluation Return", fontsize=13)
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_loss_comparison(
    results: Dict[str, Dict[str, List[float]]],
    save_path: str,
) -> None:
    """Plot critic (value) loss curves over training.

    Args:
        results: Mapping from condition name to a dict with keys
            ``steps`` and ``critic_loss``.
        save_path: File path for the output PNG.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    for idx, (name, data) in enumerate(results.items()):
        color = _PALETTE[idx % len(_PALETTE)]
        steps = np.asarray(data.get("critic_loss_steps", data["steps"]))
        loss = np.asarray(data["critic_loss"])
        ax.plot(steps, loss, label=name, color=color, linewidth=2)

    ax.set_xlabel("Environment Steps", fontsize=13)
    ax.set_ylabel("Critic Loss", fontsize=13)
    ax.set_title("Critic Loss Comparison", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_world_model_error(
    results: Dict[str, Dict[str, List[float]]],
    save_path: str,
) -> None:
    """Plot world-model dynamics and reward prediction errors.

    Args:
        results: Mapping from condition name to a dict with keys
            ``steps``, ``dynamics_loss``, and ``reward_loss``.
        save_path: File path for the output PNG.
    """
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Dynamics loss ---
    ax = axes[0]
    for idx, (name, data) in enumerate(results.items()):
        color = _PALETTE[idx % len(_PALETTE)]
        steps = np.asarray(data.get("dynamics_loss_steps", data["steps"]))
        loss = np.asarray(data["dynamics_loss"])
        ax.plot(steps, loss, label=name, color=color, linewidth=2)
    ax.set_xlabel("Environment Steps", fontsize=13)
    ax.set_ylabel("Dynamics Loss (MSE)", fontsize=13)
    ax.set_title("Dynamics Prediction Error", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # --- Reward loss ---
    ax = axes[1]
    for idx, (name, data) in enumerate(results.items()):
        color = _PALETTE[idx % len(_PALETTE)]
        steps = np.asarray(data.get("reward_loss_steps", data["steps"]))
        loss = np.asarray(data["reward_loss"])
        ax.plot(steps, loss, label=name, color=color, linewidth=2)
    ax.set_xlabel("Environment Steps", fontsize=13)
    ax.set_ylabel("Reward Loss (MSE)", fontsize=13)
    ax.set_title("Reward Prediction Error", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "World Model Prediction Errors",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
