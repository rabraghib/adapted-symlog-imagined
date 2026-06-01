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
    parser.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=1,
        help="Number of parallel jobs to run. If 1, runs sequentially with live console output.",
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
    if not args.dry_run:
        print(f"  Concurrency  : {args.jobs} job(s)")
    print("=" * 70 + "\n")

    failed: List[str] = []
    suite_start = time.time()

    if args.jobs <= 1 or args.dry_run:
        # Sequential execution
        for idx, (env, condition, seed) in enumerate(experiments, 1):
            label = f"{env} / {condition_label(condition)} / seed={seed}"
            cmd = build_command(env, condition, seed, args.experiment_dir)

            print(f"[{idx}/{total}] {label}")
            if args.dry_run:
                print(f"  CMD: {' '.join(cmd)}\n")
                continue

            run_start = time.time()
            try:
                subprocess.run(
                    cmd,
                    check=True,
                    text=True,
                    capture_output=False,
                )
            except subprocess.CalledProcessError as exc:
                print(f"  [ERROR] FAILED (exit code {exc.returncode})")
                failed.append(label)
                continue

            elapsed = time.time() - run_start
            print(f"  [OK] Done in {elapsed:.1f}s\n")
    else:
        # Parallel execution with output redirection to log files
        import os
        log_dir = os.path.join(args.experiment_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)

        print(f"Running with {args.jobs} parallel workers. Outputs redirected to {log_dir}/\n")

        pending = list(enumerate(experiments, 1))
        running = []  # List of (idx, label, popen_obj, start_time, file_handle)
        completed = 0

        while pending or running:
            # Start new processes up to the limit
            while pending and len(running) < args.jobs:
                idx, (env, condition, seed) = pending.pop(0)
                label = f"{env} / {condition_label(condition)} / seed={seed}"
                cmd = build_command(env, condition, seed, args.experiment_dir)

                log_name = f"{env}_{condition_label(condition)}_seed{seed}.log".replace("+", "_")
                log_path = os.path.join(log_dir, log_name)

                print(f"[{idx}/{total}] STARTED: {label} (logging to {log_name})")

                f_log = open(log_path, "w", encoding="utf-8")
                run_start = time.time()
                try:
                    proc = subprocess.Popen(
                        cmd,
                        stdout=f_log,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    running.append((idx, label, proc, run_start, f_log))
                except Exception as e:
                    print(f"[{idx}/{total}] [ERROR] FAILED TO LAUNCH: {label} ({e})")
                    failed.append(label)
                    f_log.close()

            # Check status of running processes
            still_running = []
            for item in running:
                idx, label, proc, run_start, f_log = item
                ret = proc.poll()
                if ret is not None:
                    # Process completed
                    f_log.close()
                    elapsed = time.time() - run_start
                    completed += 1
                    if ret == 0:
                        print(f"[{idx}/{total}] [OK] SUCCESS: {label} in {elapsed:.1f}s")
                    else:
                        print(f"[{idx}/{total}] [ERROR] FAILED: {label} (exit code {ret})")
                        failed.append(label)
                else:
                    still_running.append(item)
            running = still_running

            # Avoid busy-waiting loop
            if pending or running:
                time.sleep(0.5)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    suite_elapsed = time.time() - suite_start

    print("\n" + "=" * 70)
    print(f"  Suite completed in {suite_elapsed / 60:.1f} min")
    if failed:
        print(f"  [ERROR] {len(failed)} run(s) failed:")
        for f in failed:
            print(f"      - {f}")
    else:
        print("  [OK] All runs succeeded!")
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
