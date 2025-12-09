#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_sweep_data(filepath):
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Warning: {filepath} not found")
        return None


def create_figures(network="smallworld", data_dir=None, out_dir=None):
    network = network.lower()
    if network not in {"smallworld", "lattice"}:
        raise ValueError(f"Unsupported network '{network}'")

    data_dir = data_dir or os.path.join("data", network)
    out_dir = out_dir or os.path.join("figures", network)
    os.makedirs(out_dir, exist_ok=True)

    neighbors_file = "sweep_neighbors_per_node.csv" if network == "smallworld" else "sweep_lattice_radius.csv"
    neighbors_df = load_sweep_data(os.path.join(data_dir, neighbors_file))
    features_df = load_sweep_data(os.path.join(data_dir, "sweep_num_features.csv"))
    feature_dim_df = load_sweep_data(os.path.join(data_dir, "sweep_feature_dim.csv"))
    rewiring_df = None
    if network == "smallworld":
        rewiring_df = load_sweep_data(os.path.join(data_dir, "sweep_rewiring_prob.csv"))

    required = [neighbors_df, features_df, feature_dim_df]
    if any(df is None for df in required):
        print("Error: missing CSVs")
        return
    if network == "smallworld" and rewiring_df is None:
        print("Error: missing rewiring sweep CSV")
        return

    neighbors_df = neighbors_df.sort_values("param_value").reset_index(drop=True)
    features_df = features_df.sort_values("param_value").reset_index(drop=True)
    feature_dim_df = feature_dim_df.sort_values("param_value").reset_index(drop=True)
    if rewiring_df is not None:
        rewiring_df = rewiring_df.sort_values("param_value").reset_index(drop=True)

    # ----- Connectivity sweep -----
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    title = "Network Connectivity Effects" if network == "smallworld" else "2D Lattice Connectivity"
    fig.suptitle(title, fontsize=15, fontweight="bold")
    x_label = "Neighbors per node" if network == "smallworld" else "Lattice radius (Manhattan)"

    ax = axes[0, 0]
    ax.plot(neighbors_df["param_value"], neighbors_df["num_cultures"],
            marker="o", linewidth=2)
    ax.set_xlabel(x_label)
    ax.set_ylabel("# distinct cultures")
    ax.set_title("Fragmentation")
    ax.grid(True, linestyle="--", alpha=0.3)
    if "mean_degree" in neighbors_df.columns:
        ax_deg = ax.twinx()
        ax_deg.plot(neighbors_df["param_value"], neighbors_df["mean_degree"],
                    color="gray", linestyle="--", linewidth=1.5)
        ax_deg.set_ylabel("Mean degree", color="gray")
        ax_deg.tick_params(axis="y", colors="gray")

    ax = axes[0, 1]
    ax.plot(neighbors_df["param_value"], neighbors_df["largest_culture_fraction"],
            marker="s", linewidth=2)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Largest culture fraction")
    ax.set_title("Dominant culture size")
    ax.grid(True, linestyle="--", alpha=0.3)

    ax = axes[1, 0]
    ax.plot(neighbors_df["param_value"], neighbors_df["entropy"],
            marker="^", linewidth=2)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Entropy")
    ax.set_title("Cultural diversity")
    ax.grid(True, linestyle="--", alpha=0.3)

    ax = axes[1, 1]
    ax.plot(neighbors_df["param_value"], neighbors_df["edge_homophily"],
            marker="D", linewidth=2)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Edge homophily")
    ax.set_title("Assortativity by culture")
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "figure1_connectivity.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # ----- Rewiring sweep (small-world only) -----
    if rewiring_df is not None:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Small-World Rewiring Effects", fontsize=15, fontweight="bold")

        ax = axes[0, 0]
        ax.plot(rewiring_df["param_value"], rewiring_df["num_cultures"],
                marker="o", linewidth=2)
        ax.set_xlabel("Rewiring probability p")
        ax.set_ylabel("# distinct cultures")
        ax.set_title("Fragmentation vs disorder")
        ax.grid(True, linestyle="--", alpha=0.3)

        ax = axes[0, 1]
        ax.plot(rewiring_df["param_value"], rewiring_df["largest_culture_fraction"],
                marker="s", linewidth=2)
        ax.set_xlabel("Rewiring probability p")
        ax.set_ylabel("Largest culture fraction")
        ax.set_title("Dominant culture vs disorder")
        ax.grid(True, linestyle="--", alpha=0.3)

        ax = axes[1, 0]
        ax.plot(rewiring_df["param_value"], rewiring_df["entropy"],
                marker="^", linewidth=2)
        ax.set_xlabel("Rewiring probability p")
        ax.set_ylabel("Entropy")
        ax.set_title("Diversity vs disorder")
        ax.grid(True, linestyle="--", alpha=0.3)

        ax = axes[1, 1]
        ax.plot(rewiring_df["param_value"], rewiring_df["edge_homophily"],
                marker="D", linewidth=2)
        ax.set_xlabel("Rewiring probability p")
        ax.set_ylabel("Edge homophily")
        ax.set_title("Homophily vs disorder")
        ax.grid(True, linestyle="--", alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "figure2_rewiring.png"), dpi=300, bbox_inches="tight")
        plt.close()

    # ----- Number of features -----
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Number of Features F", fontsize=15, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(features_df["param_value"], features_df["num_cultures"],
            marker="o", linewidth=2)
    ax.set_xlabel("F")
    ax.set_ylabel("# distinct cultures")
    ax.set_title("Fragmentation vs F")
    ax.set_yscale("log")
    ax.grid(True, linestyle="--", alpha=0.3)

    ax = axes[0, 1]
    ax.plot(features_df["param_value"], features_df["largest_culture_fraction"],
            marker="s", linewidth=2)
    ax.set_xlabel("F")
    ax.set_ylabel("Largest culture fraction")
    ax.set_title("Dominant culture vs F")
    ax.grid(True, linestyle="--", alpha=0.3)

    ax = axes[1, 0]
    ax.plot(features_df["param_value"], features_df["entropy"],
            marker="^", linewidth=2)
    ax.set_xlabel("F")
    ax.set_ylabel("Entropy")
    ax.set_title("Diversity vs F")
    ax.grid(True, linestyle="--", alpha=0.3)

    ax = axes[1, 1]
    ax.plot(features_df["param_value"], features_df["fragmentation"],
            marker="D", linewidth=2)
    ax.set_xlabel("F")
    ax.set_ylabel("Fragmentation index")
    ax.set_title("Fragmentation index vs F")
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "figure3_features.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # ----- Feature dimension -----
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Feature Dimension q (values per feature)", fontsize=15, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(feature_dim_df["param_value"], feature_dim_df["num_cultures"],
            marker="o", linewidth=2)
    ax.set_xlabel("q")
    ax.set_ylabel("# distinct cultures")
    ax.set_title("Fragmentation vs q")
    ax.grid(True, linestyle="--", alpha=0.3)

    ax = axes[0, 1]
    ax.plot(feature_dim_df["param_value"], feature_dim_df["largest_culture_fraction"],
            marker="s", linewidth=2)
    ax.set_xlabel("q")
    ax.set_ylabel("Largest culture fraction")
    ax.set_title("Dominant culture vs q")
    ax.grid(True, linestyle="--", alpha=0.3)

    ax = axes[1, 0]
    ax.plot(feature_dim_df["param_value"], feature_dim_df["entropy"],
            marker="^", linewidth=2)
    ax.set_xlabel("q")
    ax.set_ylabel("Entropy")
    ax.set_title("Diversity vs q")
    ax.grid(True, linestyle="--", alpha=0.3)

    ax = axes[1, 1]
    ax.plot(feature_dim_df["param_value"], feature_dim_df["edge_homophily"],
            marker="D", linewidth=2)
    ax.set_xlabel("q")
    ax.set_ylabel("Edge homophily")
    ax.set_title("Homophily vs q")
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "figure4_feature_dim.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Comparative overview
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(f"Comparative: largest culture fraction ({network})", fontsize=14, fontweight="bold")

    series = [
        (neighbors_df, "param_value", "largest_culture_fraction",
         "neighbors" if network == "smallworld" else "radius"),
        (features_df, "param_value", "largest_culture_fraction", "F"),
        (feature_dim_df, "param_value", "largest_culture_fraction", "q"),
    ]
    if rewiring_df is not None:
        series.append((rewiring_df, "param_value", "largest_culture_fraction", "rewiring p"))

    for df, xcol, ycol, label in series:
        ax.plot(df[xcol] / df[xcol].max(), df[ycol], marker="o", linewidth=2, label=label)

    ax.set_xlabel("normalized parameter value")
    ax.set_ylabel("largest culture fraction")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "figure5_comparative.png"), dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Axelrod sweep results.")
    parser.add_argument("--network", choices=["smallworld", "lattice"], default="smallworld",
                        help="Which network results to plot (matches data/<network>).")
    parser.add_argument("--data-dir", help="Override data directory (default: data/<network>)")
    parser.add_argument("--out-dir", help="Override output directory (default: figures/<network>)")
    args = parser.parse_args()

    create_figures(network=args.network, data_dir=args.data_dir, out_dir=args.out_dir)
