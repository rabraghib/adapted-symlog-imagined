"""Post-experiment analysis and comparison.

Scans the experiment output directory for result JSON files, groups them by
condition (env + agent type + symlog flag), aggregates across random seeds,
generates comparison plots, and prints a summary table.

Usage::

    python -m src.evaluate --experiment-dir experiments
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

from src.utils.plotting import (
    plot_learning_curves,
    plot_loss_comparison,
    plot_world_model_error,
)


# ======================================================================
# Experiment name parsing
# ======================================================================

# Expected format: {env}_{agent_type}[_symlog]_seed{N}
_NAME_PATTERN = re.compile(
    r"^(?P<env>.+?)_(?P<agent>model_free|imagination)(?:_(?P<symlog>symlog))?_seed(?P<seed>\d+)$"
)


def parse_experiment_name(name: str) -> Dict[str, Any] | None:
    """Extract structured fields from an experiment directory name.

    Returns:
        Dict with keys ``env``, ``agent``, ``use_symlog`` (bool),
        ``seed`` (int), or *None* if the name does not match.
    """
    m = _NAME_PATTERN.match(name)
    if m is None:
        return None
    return {
        "env": m.group("env"),
        "agent": m.group("agent"),
        "use_symlog": m.group("symlog") is not None,
        "seed": int(m.group("seed")),
    }


def condition_key(info: Dict[str, Any]) -> str:
    """Build a human-readable condition label from parsed info."""
    label = f"{info['env']}_{info['agent']}"
    if info["use_symlog"]:
        label += "_symlog"
    return label


# ======================================================================
# Result loading
# ======================================================================


def load_results(experiment_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    """Scan *experiment_dir* and load results grouped by condition.

    Returns:
        ``{condition_label: [results_dict_per_seed, ...]}``.
    """
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    if not os.path.isdir(experiment_dir):
        print(f"[WARN] Experiment directory not found: {experiment_dir}")
        return grouped

    for entry in sorted(os.listdir(experiment_dir)):
        exp_path = os.path.join(experiment_dir, entry)
        if not os.path.isdir(exp_path):
            continue

        info = parse_experiment_name(entry)
        if info is None:
            continue

        # Logger.save() writes to metrics.json
        metrics_file = os.path.join(exp_path, "metrics.json")
        if not os.path.isfile(metrics_file):
            print(f"  [SKIP] No metrics.json in {entry}")
            continue

        with open(metrics_file, "r") as f:
            results = json.load(f)

        results["_meta"] = info
        key = condition_key(info)
        grouped[key].append(results)

    return grouped


# ======================================================================
# Aggregation helpers
# ======================================================================


def aggregate_eval_curves(
    runs: List[Dict[str, Any]],
) -> Dict[str, Any] | None:
    """Aggregate evaluation curves across seeds.

    The logger stores eval data under the ``eval_returns`` key as a list
    of dicts with ``step``, ``mean_return``, ``std_return``.

    Returns:
        Dict with ``steps``, ``returns_mean``, ``returns_std`` arrays,
        or *None* if no eval data is found.
    """
    if not runs:
        return None

    all_steps: List[np.ndarray] = []
    all_means: List[np.ndarray] = []

    for run in runs:
        evals = run.get("eval_returns", [])
        if not evals:
            continue
        steps = np.array([e["step"] for e in evals])
        means = np.array([e["mean_return"] for e in evals])
        all_steps.append(steps)
        all_means.append(means)

    if not all_steps:
        return None

    # Align to shortest eval sequence (all should have the same steps)
    min_len = min(len(s) for s in all_steps)
    steps = all_steps[0][:min_len]
    means_matrix = np.array([m[:min_len] for m in all_means])

    return {
        "steps": steps,
        "returns_mean": means_matrix.mean(axis=0),
        "returns_std": means_matrix.std(axis=0),
    }


def aggregate_loss_curves(
    runs: List[Dict[str, Any]], loss_key: str
) -> Dict[str, np.ndarray] | None:
    """Aggregate a specific loss metric across seeds.

    The logger stores training metrics under the ``train_steps`` key.

    Returns dict with ``steps`` and ``values`` arrays, or *None* if the
    metric is not present in any run.
    """
    all_steps: List[np.ndarray] = []
    all_values: List[np.ndarray] = []

    for run in runs:
        training = run.get("train_steps", [])
        if not training:
            continue
        steps = []
        values = []
        for entry in training:
            if loss_key in entry:
                steps.append(entry["step"])
                values.append(entry[loss_key])
        if steps:
            all_steps.append(np.array(steps))
            all_values.append(np.array(values))

    if not all_steps:
        return None

    # Use the shortest run for alignment
    min_len = min(len(s) for s in all_steps)
    steps = all_steps[0][:min_len]
    vals = np.array([v[:min_len] for v in all_values])
    return {"steps": steps, "values": vals.mean(axis=0)}


# ======================================================================
# Summary table
# ======================================================================


def print_summary(grouped: Dict[str, List[Dict[str, Any]]]) -> None:
    """Print a nicely-formatted summary table to the console."""
    print("\n" + "=" * 75)
    print(f"  {'Condition':<40}  {'Seeds':>5}  {'Final Mean':>11}  {'± Std':>8}")
    print("-" * 75)

    for cond, runs in sorted(grouped.items()):
        agg = aggregate_eval_curves(runs)
        if not agg:
            print(f"  {cond:<40}  {len(runs):>5}  {'N/A':>11}  {'N/A':>8}")
            continue
        final_mean = agg["returns_mean"][-1]
        final_std = agg["returns_std"][-1]
        print(
            f"  {cond:<40}  {len(runs):>5}  {final_mean:>11.1f}  {final_std:>8.1f}"
        )

    print("=" * 75 + "\n")


# ======================================================================
# Main
# ======================================================================


def main() -> None:
    """Entry point for the evaluation / plotting pipeline."""
    parser = argparse.ArgumentParser(
        description="Aggregate and plot experiment results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--experiment-dir",
        type=str,
        default="experiments",
        help="Root directory containing experiment sub-directories.",
    )
    args = parser.parse_args()

    # 1. Load results
    print(f"Scanning {args.experiment_dir} for results...")
    grouped = load_results(args.experiment_dir)

    if not grouped:
        print("No experiment results found. Exiting.")
        return

    print(f"Found {sum(len(v) for v in grouped.values())} runs across "
          f"{len(grouped)} conditions.\n")

    # 2. Summary table
    print_summary(grouped)

    # 3. Generate plots
    plot_dir = os.path.join(args.experiment_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Group conditions by environment for per-env comparison plots
    env_conditions: Dict[str, Dict[str, Dict]] = defaultdict(dict)
    for cond, runs in grouped.items():
        # Get environment name from metadata
        env_name = None
        for run in runs:
            meta = run.get("_meta", {})
            if "env" in meta:
                env_name = meta["env"]
                break
        if env_name is None:
            continue

        # Aggregate eval curves for plotting (uses keys that plotting expects)
        agg = aggregate_eval_curves(runs)
        if agg:
            cond_data = {
                "steps": agg["steps"],
                "returns_mean": agg["returns_mean"],
                "returns_std": agg["returns_std"],
            }
            # Attach loss curves if available
            for loss_key in ["critic_loss", "actor_loss"]:
                loss_agg = aggregate_loss_curves(runs, loss_key)
                if loss_agg is not None:
                    cond_data[loss_key] = loss_agg["values"]

            # World model losses
            for wm_key in ["dynamics_loss", "reward_loss"]:
                wm_agg = aggregate_loss_curves(runs, wm_key)
                if wm_agg is not None:
                    cond_data[wm_key] = wm_agg["values"]

            env_conditions[env_name][cond] = cond_data

    for env_name, env_results in env_conditions.items():
        safe_env = env_name.replace("/", "_")

        # Learning curves
        plot_learning_curves(
            env_results,
            save_path=os.path.join(plot_dir, f"{safe_env}_learning_curves.png"),
            title=f"{env_name} — Learning Curves",
        )
        print(f"  Saved learning curves for {env_name}")

        # Loss comparison (only for conditions that have critic_loss)
        loss_results = {}
        for c, d in env_results.items():
            if "critic_loss" in d:
                # plotting.plot_loss_comparison expects 'steps' and 'critic_loss'
                loss_results[c] = d
        if loss_results:
            plot_loss_comparison(
                loss_results,
                save_path=os.path.join(plot_dir, f"{safe_env}_loss_comparison.png"),
            )
            print(f"  Saved loss comparison for {env_name}")

        # World-model error (only relevant for imagination agents)
        wm_results = {
            c: d for c, d in env_results.items()
            if "dynamics_loss" in d and "reward_loss" in d
        }
        if wm_results:
            plot_world_model_error(
                wm_results,
                save_path=os.path.join(
                    plot_dir, f"{safe_env}_world_model_error.png"
                ),
            )
            print(f"  Saved world model error plot for {env_name}")

    print(f"\nAll plots saved to: {plot_dir}")


if __name__ == "__main__":
    main()
