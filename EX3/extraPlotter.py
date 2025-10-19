#!/usr/bin/env python3
"""
Run the extra perceptron experiments and plot their results.

This helper executes the compiled binary in single-mode for the extra points,
reads the CSV dumps produced by the C++ code, and generates ready-to-use plots.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "plots"
DEFAULT_BINARY = ROOT / "bin" / ("main.exe" if os.name == "nt" else "main")


def run_exercise(binary: Path, exercise_id: int, *, env: Optional[Dict[str, str]] = None) -> None:
    """Execute the compiled binary for a specific exercise slot."""
    cmd = [str(binary), "single", str(exercise_id)]
    print(f"[extraPlotter] Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"Execution failed with exit code {exc.returncode}") from exc


def load_extra_one(csv_path: Path) -> List[Dict[str, float]]:
    """Load sigma sweep results."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing data file: {csv_path}")
    rows: List[Dict[str, float]] = []
    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                rows.append(
                    {
                        "sigma": float(row["sigma"]),
                        "train_size": float(row["train_size"]),
                        "test_size": float(row["test_size"]),
                        "trials": float(row["trials"]),
                        "mean_accuracy": float(row["mean_accuracy"]),
                        "std_accuracy": float(row["std_accuracy"]),
                        "mean_epochs": float(row["mean_epochs"]),
                        "std_epochs": float(row["std_epochs"]),
                        "mean_errors": float(row["mean_errors"]),
                        "std_errors": float(row["std_errors"]),
                    }
                )
            except KeyError as exc:
                raise ValueError(
                    "extra_point_one.csv is missing expected columns. "
                    "Please rerun the extra experiments to regenerate the file."
                ) from exc
    rows.sort(key=lambda item: item["sigma"])
    return rows


def load_extra_two(csv_path: Path) -> Dict[int, List[Tuple[int, float, float]]]:
    """Load grouped accuracy curves for each bit-width."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing data file: {csv_path}")
    grouped: Dict[int, List[Tuple[int, float, float]]] = defaultdict(list)
    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            bits = int(row["bits"])
            dataset_size = int(row["dataset_size"])
            mean_acc = float(row["mean_accuracy"])
            std_acc = float(row["std_accuracy"])
            grouped[bits].append((dataset_size, mean_acc, std_acc))
    for rows in grouped.values():
        rows.sort(key=lambda item: item[0])
    return grouped


def plot_extra_one(data: List[Dict[str, float]], output_path: Path, show: bool) -> None:
    """Plot accuracy and learning dynamics as a function of noise sigma."""
    if not data:
        raise ValueError("No data available for extra point 1.")

    sigma = np.array([row["sigma"] for row in data], dtype=float)
    mean_accuracy = np.array([row["mean_accuracy"] for row in data], dtype=float)
    std_accuracy = np.array([row["std_accuracy"] for row in data], dtype=float)
    mean_epochs = np.array([row["mean_epochs"] for row in data], dtype=float)
    std_epochs = np.array([row["std_epochs"] for row in data], dtype=float)
    mean_errors = np.array([row["mean_errors"] for row in data], dtype=float)
    std_errors = np.array([row["std_errors"] for row in data], dtype=float)

    train_size = int(round(data[0]["train_size"]))
    test_size = int(round(data[0]["test_size"]))
    trials = int(round(data[0]["trials"]))
    
    # Convert standard deviations to standard errors (divide by sqrt(trials))
    sqrt_trials = np.sqrt(trials)
    std_accuracy = std_accuracy / sqrt_trials
    std_epochs = std_epochs / sqrt_trials
    std_errors = std_errors / sqrt_trials

    for row in data[1:]:
        if (
            int(round(row["train_size"])) != train_size
            or int(round(row["test_size"])) != test_size
            or int(round(row["trials"])) != trials
        ):
            raise ValueError(
                "Inconsistent sigma sweep configuration detected. "
                f"Expected train={train_size}, test={test_size}, trials={trials}."
            )

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))
    ax_accuracy, ax_dynamics = axes

    lower_acc = np.maximum(mean_accuracy - std_accuracy, 0.0)
    upper_acc = np.minimum(mean_accuracy + std_accuracy, 100.0)

    ax_accuracy.plot(sigma, mean_accuracy, marker="o", color="tab:blue")
    ax_accuracy.fill_between(
        sigma, lower_acc, upper_acc, alpha=0.2, color="tab:blue"
    )
    ax_accuracy.set_ylabel("Accuracy (%)")
    ax_accuracy.set_title(f"Accuracy vs Noise (test size={test_size})")
    ax_accuracy.set_xlabel("Noise std dev (sigma)")
    # ax_accuracy.set_ylim(0, 100)
    ax_accuracy.set_xscale("log")
    ax_accuracy.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    lower_epochs = np.maximum(mean_epochs - std_epochs, 0.0)
    upper_epochs = mean_epochs + std_epochs
    ax_dynamics.plot(sigma, mean_epochs, marker="s", color="tab:orange", label="Epochs run")
    ax_dynamics.fill_between(sigma, lower_epochs, upper_epochs, alpha=0.2, color="tab:orange")

    lower_errors = np.maximum(mean_errors - std_errors, 0.0)
    upper_errors = mean_errors + std_errors
    ax_dynamics.plot(sigma, mean_errors, marker="^", color="tab:green", label="Residual errors")
    ax_dynamics.fill_between(sigma, lower_errors, upper_errors, alpha=0.2, color="tab:green")
    ax_dynamics.set_xlabel("Noise std dev (sigma)")
    ax_dynamics.set_ylabel("Count")
    ax_dynamics.set_title(f"Training dynamics (train size={train_size}, trials={trials})")
    ax_dynamics.set_xscale("log")
    ax_dynamics.legend()
    ax_dynamics.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    fig.suptitle(
        f"Extra Point 1: Noise sweep (train={train_size}, test={test_size}, trials={trials})",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_extra_two(grouped: Dict[int, List[Tuple[int, float, float]]], output_path: Path, show: bool) -> None:
    """Plot accuracy curves for the different bit-width configurations, raw and rescaled."""
    # Extra point 2 uses 200 trials (hardcoded in C++ extraPointTwo function)
    num_trials = 200
    sqrt_trials = np.sqrt(num_trials)
    
    processed: Dict[int, Dict[str, np.ndarray]] = {}
    max_alphas: List[float] = []
    for bits, rows in grouped.items():
        sizes = np.array([item[0] for item in rows], dtype=float)
        mean = np.array([item[1] for item in rows], dtype=float)
        std = np.array([item[2] for item in rows], dtype=float)
        # Convert standard deviation to standard error
        std = std / sqrt_trials
        n_weights = float(bits * 2)
        alpha = sizes / n_weights
        max_alphas.append(float(np.max(alpha)))
        processed[bits] = {
            "sizes": sizes,
            "mean": mean,
            "std": std,
            "alpha": alpha,
            "n_weights": n_weights,
        }

    if not processed:
        raise ValueError("No data available for extra point 2.")

    max_shared_alpha = min(max_alphas)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), sharey=True)
    ax_raw, ax_rescaled = axes

    for bits in sorted(processed):
        data = processed[bits]
        mask = data["alpha"] <= (max_shared_alpha + 1e-9)
        sizes = data["sizes"][mask]
        mean = data["mean"][mask]
        std = data["std"][mask]
        alpha = data["alpha"][mask]

        ax_raw.plot(sizes, mean, marker="o", label=f"{bits} bits")
        ax_raw.fill_between(sizes, mean - std, mean + std, alpha=0.15)

        order = np.argsort(alpha)
        alpha_sorted = alpha[order]
        mean_sorted = mean[order]
        std_sorted = std[order]

        ax_rescaled.plot(alpha_sorted, mean_sorted, marker="o", label=f"{bits} bits")
        ax_rescaled.fill_between(alpha_sorted, mean_sorted - std_sorted, mean_sorted + std_sorted, alpha=0.15)

    ax_raw.set_xlabel("Training set size (P)")
    ax_raw.set_ylabel("Accuracy (%)")
    ax_raw.set_title("Accuracy vs. Dataset Size")
    ax_raw.set_ylim(0, 100)
    ax_raw.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax_raw.legend()

    ax_rescaled.set_xlabel(r"Patterns per weight $\alpha = P/N$")
    ax_rescaled.set_title("Accuracy vs. Rescaled Load")
    ax_rescaled.set_ylim(0, 100)
    ax_rescaled.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax_rescaled.legend()

    fig.suptitle("Extra Point 2 · Training performance across bit widths", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run extra experiments and generate plots.")
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY, help="Path to the compiled executable.")
    parser.add_argument("--skip-run", action="store_true", help="Reuse existing CSV data without rerunning the executable.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATA_DIR, help="Directory for output figures.")
    parser.add_argument("--no-show", action="store_true", help="Do not display the figures interactively.")
    parser.add_argument(
        "--sigma",
        type=float,
        action="append",
        dest="sigma_list",
        help="Noise sigma to evaluate for extra point 1 (repeatable). Defaults to built-in sweep.",
    )
    parser.add_argument(
        "--extra",
        choices=["1", "2", "both"],
        default="both",
        help="Which extra point(s) to run/plot: '1', '2', or 'both' (default: both).",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=2000,
        help="Number of independent trials per sigma for extra point 1.",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=100,
        help="Training set size per trial for extra point 1.",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=2000,
        help="Test set size per trial for extra point 1.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    binary = args.binary

    if not binary.exists():
        raise SystemExit(f"Executable '{binary}' not found. Build the project first.")

    data_dir = DEFAULT_DATA_DIR
    default_sigmas = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, math.sqrt(50.0), 10.0,15.0]
    sigma_list = args.sigma_list if args.sigma_list else default_sigmas

    # Decide which extra points to run/plot based on the --extra flag
    extra_choice = args.extra

    if not args.skip_run:
        if extra_choice in ("1", "both"):
            env_extra_one = os.environ.copy()
            env_extra_one["EXTRA_SIGMAS"] = ",".join(str(s) for s in sigma_list)
            env_extra_one["EXTRA_TRIALS"] = str(args.trials)
            env_extra_one["EXTRA_TRAIN_SIZE"] = str(args.train_size)
            env_extra_one["EXTRA_TEST_SIZE"] = str(args.test_size)
            run_exercise(binary, 11, env=env_extra_one)

        if extra_choice in ("2", "both"):
            run_exercise(binary, 12)

    figures_dir = args.output_dir
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load and plot only the requested extras. If CSV data is missing, warn and skip.
    if extra_choice in ("1", "both"):
        csv_one = data_dir / "extra_point_one.csv"
        if csv_one.exists():
            extra_one_data = load_extra_one(csv_one)
            plot_extra_one(extra_one_data, figures_dir / "extra_point_one.png", show=not args.no_show)
        else:
            print(f"[extraPlotter] Warning: '{csv_one}' not found. Skipping Extra Point 1.")

    if extra_choice in ("2", "both"):
        csv_two = data_dir / "extra_point_two.csv"
        if csv_two.exists():
            extra_two_data = load_extra_two(csv_two)
            plot_extra_two(extra_two_data, figures_dir / "extra_point_two.png", show=not args.no_show)
        else:
            print(f"[extraPlotter] Warning: '{csv_two}' not found. Skipping Extra Point 2.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI surface
        print(f"[extraPlotter] Error: {exc}", file=sys.stderr)
        sys.exit(1)
