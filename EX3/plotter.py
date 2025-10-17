#!/usr/bin/env python3
"""
Exercise Plotters

Usage:
  python exercise_plotters.py --exercise 8
  python exercise_plotters.py --all
"""

from dataclasses import dataclass
from typing import List, Dict, Callable, Optional
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# ---------- Base container ----------
@dataclass
class AccuracyCurve:
    dataset_sizes: List[int]
    mean_accuracy: List[float]
    std_accuracy: List[float]
    label: str

# ---------- Exercise 7 ----------
class ExerciseSevenPlotter:
    @staticmethod
    def data() -> AccuracyCurve:
        return AccuracyCurve(
            dataset_sizes=[1, 10, 20, 50, 100, 150, 200],
            mean_accuracy=[50.95, 57.86, 65.27, 78.90, 88.16, 91.86, 93.87],
            std_accuracy=[7.63, 7.00, 6.21, 4.55, 3.00, 2.32, 1.80],
            label="20 bits"
        )

    @staticmethod
    def plot(show: bool = True, save_path: Optional[str] = None) -> None:
        d = ExerciseSevenPlotter.data()
        x = np.array(d.dataset_sizes)
        mu = np.array(d.mean_accuracy)
        sigma = np.array(d.std_accuracy)

        plt.figure()
        plt.plot(x, mu, marker="o", label=d.label)
        plt.fill_between(x, mu - sigma, mu + sigma, alpha=0.2)

        plt.title("Exercise 7 — Accuracy vs Dataset Size")
        plt.xlabel("Dataset size")
        plt.ylabel("Accuracy (%)")
        plt.ylim(0, 100)
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.legend()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close()

# ---------- Exercise 8 ----------
class ExerciseEightPlotter:
    @staticmethod
    def data_20bits() -> AccuracyCurve:
        # from Exercise 6
        return AccuracyCurve(
            dataset_sizes=[1, 10, 20, 50, 100, 150, 200],
            mean_accuracy=[50.95, 57.86, 65.27, 78.90, 88.16, 91.86, 93.87],
            std_accuracy=[7.63, 7.00, 6.21, 4.55, 3.00, 2.32, 1.80],
            label="20 bits"
        )

    @staticmethod
    def data_40bits() -> AccuracyCurve:
        # your new dataset
        return AccuracyCurve(
            dataset_sizes=[1, 2, 20, 40, 100, 200, 300],
            mean_accuracy=[50.48, 50.98, 58.57, 65.13, 78.70, 87.82, 91.75],
            std_accuracy=[5.23, 5.62, 4.87, 4.66, 3.31, 2.32, 1.73],
            label="40 bits"
        )

    @staticmethod
    def plot(show: bool = True, save_path: Optional[str] = None) -> None:
        d20 = ExerciseEightPlotter.data_20bits()
        d40 = ExerciseEightPlotter.data_40bits()

        plt.figure()

        for d in [d20, d40]:
            x = np.array(d.dataset_sizes)
            mu = np.array(d.mean_accuracy)
            sigma = np.array(d.std_accuracy)
            plt.plot(x, mu, marker="o", label=d.label)
            plt.fill_between(x, mu - sigma, mu + sigma, alpha=0.2)

        plt.title("Exercise 8 — Accuracy vs Dataset Size (20 bits vs 40 bits)")
        plt.xlabel("Dataset size")
        plt.ylabel("Accuracy (%)")
        plt.ylim(0, 100)
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.legend()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close()

# ---------- Registry ----------
PLOTTERS: Dict[int, Callable[..., None]] = {
    7: ExerciseSevenPlotter.plot,
    8: ExerciseEightPlotter.plot,
}

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser()
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--exercise", type=int)
    group.add_argument("--all", action="store_true")
    p.add_argument("--save-dir", type=str, default=None)
    p.add_argument("--no-show", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    if args.all:
        targets = sorted(PLOTTERS.items())
    else:
        if args.exercise not in PLOTTERS:
            raise SystemExit(f"No plotter for exercise {args.exercise}.")
        targets = [(args.exercise, PLOTTERS[args.exercise])]

    for ex_num, plotter in targets:
        save_path = None
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            save_path = os.path.join(args.save_dir, f"exercise_{ex_num}.png")
        plotter(show=not args.no_show, save_path=save_path)

if __name__ == "__main__":
    main()
