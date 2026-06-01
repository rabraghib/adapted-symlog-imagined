"""Training metrics logger.

Collects per-step loss metrics and periodic evaluation returns, then
persists them to disk as JSON for downstream analysis and plotting.
"""

import json
import os
from dataclasses import asdict
from typing import Any, Dict, List

from src.config import Config


class Logger:
    """Simple JSON-based metrics logger.

    On construction the logger creates the output directory (if it does
    not exist) and serialises the run configuration.  Metrics are
    accumulated in memory and can be flushed to disk at any time via
    :meth:`save`.

    Args:
        log_dir: Directory in which to write log files.
        config: Experiment configuration to persist alongside metrics.
    """

    def __init__(self, log_dir: str, config: Config) -> None:
        self.log_dir = log_dir
        self.config = config

        os.makedirs(log_dir, exist_ok=True)

        # Persist the configuration for reproducibility.
        config_path = os.path.join(log_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(asdict(config), f, indent=2)

        # Accumulated metrics.
        self.train_steps: List[Dict[str, Any]] = []
        self.eval_steps: List[int] = []
        self.eval_returns: List[Dict[str, float]] = []
        self.losses: List[Dict[str, Any]] = []

    def log_step(self, step: int, metrics: Dict[str, Any]) -> None:
        """Record training metrics for a given step.

        Args:
            step: Current environment step counter.
            metrics: Arbitrary key-value pairs (e.g. loss components).
        """
        entry = {"step": step, **metrics}
        self.train_steps.append(entry)
        self.losses.append(entry)

    def log_eval(
        self, step: int, mean_return: float, std_return: float
    ) -> None:
        """Record an evaluation result and print a summary.

        Args:
            step: Current environment step counter.
            mean_return: Mean episodic return across evaluation episodes.
            std_return: Standard deviation of episodic returns.
        """
        self.eval_steps.append(step)
        self.eval_returns.append(
            {"step": step, "mean_return": mean_return, "std_return": std_return}
        )
        print(
            f"Step {step} | Eval Return: {mean_return:.1f} "
            f"\u00b1 {std_return:.1f}"
        )

    def save(self) -> None:
        """Flush all accumulated metrics to a JSON file on disk."""
        metrics_path = os.path.join(self.log_dir, "metrics.json")
        data = {
            "train_steps": self.train_steps,
            "eval_steps": self.eval_steps,
            "eval_returns": self.eval_returns,
            "losses": self.losses,
        }
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def get_results(self) -> Dict[str, Any]:
        """Return all logged data as a dictionary.

        Returns:
            Dictionary with keys ``train_steps``, ``eval_steps``,
            ``eval_returns``, and ``losses``.
        """
        return {
            "train_steps": self.train_steps,
            "eval_steps": self.eval_steps,
            "eval_returns": self.eval_returns,
            "losses": self.losses,
        }
