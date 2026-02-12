#!/usr/bin/env python3
"""
plot_innovations.py
Select innovation couples discovered after 2010 and plot their
cosine-similarity trajectory across years with null-model context.

Generates 4 figures:
  - Fig 1: 3x3 grid (9 couples)
  - Fig 2-4: 3x1 grids (3 couples each)

Each subplot shows:
  - Cosine similarity (blue line, clamped to [0,1])
  - Discovery year (red vertical dashed line)
  - Null-model average μ (orange line)
  - Null-model μ ± σ band (purple shaded area)
  - Human-readable MeSH descriptions in the title
"""

import sys
import csv
import json
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

FILTERED_CSV = "output/null_model_filtered.csv"
DATASET_JSONL = "Dataset.jsonl"

# Parse command line arguments
DARKMODE = "--darkmode" in sys.argv

# Configure styling based on mode
if DARKMODE:
    sns.set_style("darkgrid")
    sns.set_palette("husl")
    plt.rcParams['figure.facecolor'] = 'black'
    plt.rcParams['axes.facecolor'] = 'black'
    plt.rcParams['axes.edgecolor'] = 'white'
    plt.rcParams['text.color'] = 'white'
    plt.rcParams['axes.labelcolor'] = 'white'
    plt.rcParams['xtick.color'] = 'white'
    plt.rcParams['ytick.color'] = 'white'
    plt.rcParams['legend.facecolor'] = 'black'
    plt.rcParams['legend.edgecolor'] = 'white'
    plt.rcParams['grid.color'] = 'gray'
    FACECOLOR = 'black'
    TITLE_COLOR = 'white'
    LABEL_COLORS = {
        'mu_band': 'lightgray',
        'mu_line': 'lightgray',
        'cosine_line': 'white',
        'discovery_line': 'yellow',
    }
else:
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rcParams['text.color'] = 'black'
    plt.rcParams['axes.labelcolor'] = 'black'
    plt.rcParams['xtick.color'] = 'black'
    plt.rcParams['ytick.color'] = 'black'
    plt.rcParams['legend.facecolor'] = 'white'
    plt.rcParams['legend.edgecolor'] = 'black'
    plt.rcParams['grid.color'] = 'lightgray'
    FACECOLOR = 'white'
    TITLE_COLOR = '#2c3e50'
    LABEL_COLORS = {
        'mu_band': 'mediumpurple',
        'mu_line': 'indigo',
        'cosine_line': 'dodgerblue',
        'discovery_line': 'crimson',
    }

# ─── Build MeSH code → human name lookup from Dataset.jsonl ─────────────────

def build_mesh_lookup(jsonl_path):
    """Scan the dataset and return a dict mapping 'D003267' → 'Contraception'."""
    lookup = {}
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for m in obj.get("mesh", []):
                    ui = m.get("ui", "")
                    name = m.get("name", "")
                    if ui and name and ui not in lookup:
                        lookup[ui] = name
    except FileNotFoundError:
        print(f"Warning: {jsonl_path} not found – will use raw codes")
    return lookup


def main():
    # ── Build MeSH description lookup ───────────────────────────────────
    print("Building MeSH description lookup...")
    mesh_names = build_mesh_lookup(DATASET_JSONL)
    print(f"  {len(mesh_names)} MeSH codes resolved")

    # ── Load filtered CSV ───────────────────────────────────────────────
    try:
        with open(FILTERED_CSV, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except FileNotFoundError:
        print(f"Error: {FILTERED_CSV} not found. Run  bin/main nullmodel  first.")
        sys.exit(1)

    print(f"Loaded {len(rows)} filtered innovation pairs")

    # ── Filter to discoveries after 2010 ────────────────────────────────
    rows_post2010 = [r for r in rows if int(r["discovery_year"]) > 2010]
    print(f"Filtered to {len(rows_post2010)} innovations discovered after 2010")

    if len(rows_post2010) < 3:
        print("Not enough post-2010 innovations to plot.")
        sys.exit(1)

    # Identify column groups
    all_cols = list(rows[0].keys())
    dot_cols   = [c for c in all_cols if c.startswith("dot_")]
    mu_cols    = [c for c in all_cols if c.startswith("mu_")]
    sigma_cols = [c for c in all_cols if c.startswith("sigma_")]
    years = [int(c.split("_")[1]) for c in dot_cols]

    # ── Helper: resolve code to human name ──────────────────────────────
    def code_label(code_str):
        """e.g. 'D003267' → 'Contraception (D003267)' or just 'D003267'."""
        name = mesh_names.get(code_str)
        if name:
            # Truncate long names to fit on single line
            if len(name) > 30:
                name = name[:27] + "..."
            return f"{name} ({code_str})"
        return code_str

    # ── Helper: plot a single subplot ───────────────────────────────────
    def plot_couple(ax, row, dot_cols, mu_cols, sigma_cols, years, mesh_names):
        """Plot a single innovation couple on the given axis."""
        # Extract raw cosine similarities and null model stats
        dots = [float(row[c]) for c in dot_cols]
        mus  = [float(row[c]) for c in mu_cols]
        sigs = [float(row[c]) for c in sigma_cols]
        disc = int(row["discovery_year"])
        z    = float(row["z_score"])

        # Null model band (μ ± σ)
        mu_lo = [m - s for m, s in zip(mus, sigs)]
        mu_hi = [m + s for m, s in zip(mus, sigs)]

        # Null model band
        if DARKMODE:
            ax.fill_between(years, mu_lo, mu_hi,
                            color=LABEL_COLORS['mu_band'], alpha=0.25, label="Null μ ± σ")
        else:
            ax.fill_between(years, mu_lo, mu_hi,
                            color=LABEL_COLORS['mu_band'], alpha=0.35, label="Null μ ± σ")

        # Null model average
        ax.plot(years, mus, "-", color=LABEL_COLORS['mu_line'], linewidth=1.5,
                alpha=0.7 if DARKMODE else 0.8, label="Null μ")

        # Cosine similarity
        ax.plot(years, dots, "o-", color=LABEL_COLORS['cosine_line'],
                markersize=5, linewidth=2.5 if DARKMODE else 2, label="Cosine sim", zorder=3)

        # Discovery year
        ax.axvline(x=disc, color=LABEL_COLORS['discovery_line'], linestyle="--", linewidth=2.5,
                   alpha=0.8, label=f"Discovery {disc}", zorder=2)

        # Title with human names (single line for codes)
        labelA = code_label(row["codeA"])
        labelB = code_label(row["codeB"])
        ax.set_title(f"{labelA} ↔ {labelB} (z = {z:.1f})",
                     fontsize=9, fontweight="bold", color=TITLE_COLOR)

        ax.legend(fontsize=7, loc="best", framealpha=0.95)
        ax.set_xlabel("Year", fontsize=9, fontweight="bold", color=TITLE_COLOR if DARKMODE else 'black')
        ax.set_ylabel("Cosine Similarity", fontsize=9, fontweight="bold", color=TITLE_COLOR if DARKMODE else 'black')
        ax.set_ylim(-0.2, 1.0)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.grid(True, alpha=0.3, linestyle="--", color='gray' if DARKMODE else 'lightgray')
        ax.set_xticks(years[::2])
        ax.tick_params(labelsize=8, colors='white' if DARKMODE else 'black')

    # ── Sample and prepare data ─────────────────────────────────────────
    # Ensure we have enough for 4 figures (9 + 3 + 3 + 3 = 18)
    min_needed = min(18, len(rows_post2010))
    if len(rows_post2010) >= 18:
        sampled = random.sample(rows_post2010, 18)
    else:
        sampled = rows_post2010
        print(f"Only {len(sampled)} post-2010 innovations available (need 18 for 4 figures)")

    # Split into groups
    group_3x3 = sampled[:9] if len(sampled) >= 9 else sampled[:min(9, len(sampled))]
    group_3x1_1 = sampled[9:12] if len(sampled) >= 12 else []
    group_3x1_2 = sampled[12:15] if len(sampled) >= 15 else []
    group_3x1_3 = sampled[15:18] if len(sampled) >= 18 else []

    # ── Figure 1: 3x3 grid ──────────────────────────────────────────────
    if len(group_3x3) > 0:
        fig1, axes1 = plt.subplots(3, 3, figsize=(18, 12))
        fig1.suptitle(
            "Innovation Prediction — Post-2010 Discoveries (3×3 Grid)",
            fontsize=16, fontweight="bold", y=0.995, color=TITLE_COLOR
        )
        for idx, row in enumerate(group_3x3):
            ax = axes1[idx // 3][idx % 3]
            plot_couple(ax, row, dot_cols, mu_cols, sigma_cols, years, mesh_names)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        out1 = "output/innovation_plots_1_grid3x3.png"
        plt.savefig(out1, dpi=150, bbox_inches="tight", facecolor=FACECOLOR)
        print(f"Saved {out1}")
        plt.close(fig1)

    # ── Figure 2: 1x3 grid (first 3 couples) ────────────────────────
    if len(group_3x1_1) > 0:
        fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
        fig2.suptitle(
            "Innovation Prediction — Post-2010 Discoveries (Selection 1)",
            fontsize=14, fontweight="bold", y=1.00, color=TITLE_COLOR
        )
        for idx, row in enumerate(group_3x1_1):
            plot_couple(axes2[idx], row, dot_cols, mu_cols, sigma_cols, years,
                       mesh_names)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        out2 = "output/innovation_plots_2_selection1.png"
        plt.savefig(out2, dpi=150, bbox_inches="tight", facecolor=FACECOLOR)
        print(f"Saved {out2}")
        plt.close(fig2)

    # ── Figure 3: 1x3 grid (second 3 couples) ───────────────────────
    if len(group_3x1_2) > 0:
        fig3, axes3 = plt.subplots(1, 3, figsize=(18, 5))
        fig3.suptitle(
            "Innovation Prediction — Post-2010 Discoveries (Selection 2)",
            fontsize=14, fontweight="bold", y=1.00, color=TITLE_COLOR
        )
        for idx, row in enumerate(group_3x1_2):
            plot_couple(axes3[idx], row, dot_cols, mu_cols, sigma_cols, years,
                       mesh_names)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        out3 = "output/innovation_plots_3_selection2.png"
        plt.savefig(out3, dpi=150, bbox_inches="tight", facecolor=FACECOLOR)
        print(f"Saved {out3}")
        plt.close(fig3)

    # ── Figure 4: 1x3 grid (third 3 couples) ────────────────────────
    if len(group_3x1_3) > 0:
        fig4, axes4 = plt.subplots(1, 3, figsize=(18, 5))
        fig4.suptitle(
            "Innovation Prediction — Post-2010 Discoveries (Selection 3)",
            fontsize=14, fontweight="bold", y=1.00, color=TITLE_COLOR
        )
        for idx, row in enumerate(group_3x1_3):
            plot_couple(axes4[idx], row, dot_cols, mu_cols, sigma_cols, years,
                       mesh_names)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        out4 = "output/innovation_plots_4_selection3.png"
        plt.savefig(out4, dpi=150, bbox_inches="tight", facecolor=FACECOLOR)
        print(f"Saved {out4}")
        plt.close(fig4)

    print("\nAll figures generated successfully!")


if __name__ == "__main__":
    main()

