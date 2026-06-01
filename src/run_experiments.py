"""Automation script to run all experimental conditions.

Launches every combination of environment, agent configuration, and seed as
a separate ``train.py`` subprocess, then generates comparison plots by
invoking ``evaluate.py``.

Usage::

    python -m src.run_experiments
    python -m src.run_experiments --experiment-dir my_experiments
    python -m src.run_experiments --dry-run  # preview commands without running
"""

from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
import time
from typing import Dict, List

# ======================================================================
# Experimental grid
# ======================================================================

ENVS: List[str] = ["CartPole-v1", "LunarLander-v3"]

CONDITIONS: List[Dict[str, object]] = [
    {"agent": "model_free", "use_symlog": False},
    {"agent": "imagination", "use_symlog": False},
    {"agent": "imagination", "use_symlog": True},
]

SEEDS: List[int] = [0, 1, 2, 3, 4]

# Per-environment step budgets
STEPS: Dict[str, int] = {
    "CartPole-v1": 100_000,
    "LunarLander-v3": 300_000,
}

DEFAULT_STEPS: int = 100_000


# ======================================================================
# Helpers
# ======================================================================


def build_command(
    env: str,
    condition: Dict[str, object],
    seed: int,
    experiment_dir: str,
) -> List[str]:
    """Build the CLI command list for a single training run."""
    total_steps = STEPS.get(env, DEFAULT_STEPS)
    cmd = [
        sys.executable,
        "-m",
        "src.train",
        "--env",
        env,
        "--agent",
        str(condition["agent"]),
        "--seed",
        str(seed),
        "--total-steps",
        str(total_steps),
        "--experiment-dir",
        experiment_dir,
    ]
    if condition.get("use_symlog"):
        cmd.append("--use-symlog")
    return cmd


def condition_label(condition: Dict[str, object]) -> str:
    """Human-readable label for a condition."""
    label = str(condition["agent"])
    if condition.get("use_symlog"):
        label += "+symlog"
    return label


# ======================================================================
# Main
# ======================================================================


def main() -> None:
    """Parse CLI args and run the full experimental grid."""
    parser = argparse.ArgumentParser(
        description="Run all experimental conditions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--experiment-dir",
        type=str,
        default="experiments",
        help="Root directory for experiment outputs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print commands without executing them.",
    )
    args = parser.parse_args()

    # Build the full list of experiments
    experiments = list(itertools.product(ENVS, CONDITIONS, SEEDS))
    total = len(experiments)

    print("=" * 70)
    print(f"  DRL Experiment Suite")
    print(f"  Environments : {', '.join(ENVS)}")
    print(f"  Conditions   : {len(CONDITIONS)}")
    print(f"  Seeds        : {len(SEEDS)}")
    print(f"  Total runs   : {total}")
    print("=" * 70 + "\n")

    failed: List[str] = []
    suite_start = time.time()

    for idx, (env, condition, seed) in enumerate(experiments, 1):
        label = f"{env} / {condition_label(condition)} / seed={seed}"
        cmd = build_command(env, condition, seed, args.experiment_dir)

        print(f"[{idx}/{total}] {label}")
        if args.dry_run:
            print(f"  CMD: {' '.join(cmd)}\n")
            continue

        run_start = time.time()
        try:
            result = subprocess.run(
                cmd,
                check=True,
                text=True,
                capture_output=False,
            )
        except subprocess.CalledProcessError as exc:
            print(f"  ✗ FAILED (exit code {exc.returncode})")
            failed.append(label)
            continue

        elapsed = time.time() - run_start
        print(f"  ✓ Done in {elapsed:.1f}s\n")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    suite_elapsed = time.time() - suite_start

    print("\n" + "=" * 70)
    print(f"  Suite completed in {suite_elapsed / 60:.1f} min")
    if failed:
        print(f"  ✗ {len(failed)} run(s) failed:")
        for f in failed:
            print(f"      - {f}")
    else:
        print("  ✓ All runs succeeded!")
    print("=" * 70 + "\n")

    if args.dry_run:
        print("(Dry run — skipping evaluation.)")
        return

    # ------------------------------------------------------------------
    # Run evaluation / plotting
    # ------------------------------------------------------------------
    print("Generating comparison plots...")
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "src.evaluate",
                "--experiment-dir",
                args.experiment_dir,
            ],
            check=True,
            text=True,
            capture_output=False,
        )
    except subprocess.CalledProcessError:
        print("[WARN] Evaluation script failed. Plots may be incomplete.")

    print("All done!")


if __name__ == "__main__":
    main()
