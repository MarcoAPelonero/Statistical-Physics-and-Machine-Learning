import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_table(path: str) -> pd.DataFrame:
    cols = ["value", "mean_steps", "sd_steps", "mean_interactions", "sd_interactions"]
    df = pd.read_csv(path, sep=r"\s+", comment="#", header=None, names=cols)
    if df.empty:
        raise ValueError(f"No data in {path}")
    return df


def plot_curves(df, ax, title, xlabel):
    x = df["value"].values
    y = df["mean_steps"].values
    
    # Linear fit: y = m*x + b
    coeffs = np.polyfit(x, y, 1)
    m, b = coeffs
    fit_line = np.poly1d(coeffs)
    x_fit = np.linspace(x.min(), x.max(), 100)
    
    ax.errorbar(
        df["value"],
        df["mean_steps"],
        yerr=df["sd_steps"],
        marker="o",
        linestyle="-",
        color="tab:blue",
        ecolor="tab:gray",
        capsize=3,
        label="Data",
    )
    ax.plot(x_fit, fit_line(x_fit), "r--", linewidth=2, label="Linear fit")
    ax.set_title(f"{title}\nFit: m = {m:.2f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Steps to convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)


def main():
    parser = argparse.ArgumentParser(
        description="Plot convergence sweeps (fragmentation < 0.05) for a 2D lattice with q=3."
    )
    parser.add_argument(
        "--data-dir",
        default="data/lattice_convergence",
        help="Directory containing convergence_num_nodes.txt, convergence_num_features.txt, convergence_radius.txt",
    )
    parser.add_argument(
        "--output",
        default="figures/convergence_times.png",
        help="Where to save the resulting figure.",
    )
    args = parser.parse_args()

    nodes_path = os.path.join(args.data_dir, "convergence_num_nodes.txt")
    features_path = os.path.join(args.data_dir, "convergence_num_features.txt")
    radius_path = os.path.join(args.data_dir, "convergence_radius.txt")

    df_nodes = load_table(nodes_path)
    df_features = load_table(features_path)
    df_radius = load_table(radius_path)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    plot_curves(df_nodes, axes[0], "Sweep: num_nodes (q=3)", "num_nodes")
    plot_curves(df_features, axes[1], "Sweep: num_features (q=3)", "num_features")
    plot_curves(df_radius, axes[2], "Sweep: lattice_radius (q=3)", "lattice_radius")

    fig.suptitle("Convergence sweeps — fragmentation threshold 0.05", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
