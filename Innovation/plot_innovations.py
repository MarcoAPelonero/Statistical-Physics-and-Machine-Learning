#!/usr/bin/env python3
"""
plot_innovations.py
Select 9 random discovered MeSH-code pairs and plot their
cosine-similarity trajectory across years, marking the discovery year.
"""

import sys
import csv
import random
import matplotlib
matplotlib.use("Agg")           # non-interactive backend (works everywhere)
import matplotlib.pyplot as plt

CSV_FILE = "output/innovations_discovered.csv"

def main():
    # ── Load CSV manually (no pandas needed) ────────────────────────────
    try:
        with open(CSV_FILE, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except FileNotFoundError:
        print(f"Error: {CSV_FILE} not found. Run  bin/main innovate  first.")
        sys.exit(1)

    print(f"Loaded {len(rows)} discovered pairs from {CSV_FILE}")

    # Identify year columns
    year_cols = [c for c in rows[0].keys() if c.startswith("dot_")]
    years     = [int(c.split("_")[1]) for c in year_cols]

    if len(rows) < 9:
        print(f"Only {len(rows)} discovered pairs available – need at least 9.")
        sys.exit(1)

    # ── Sample 9 random pairs ───────────────────────────────────────────
    
    sample = random.sample(rows, 9)

    # ── Plot ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    fig.suptitle(
        "Innovation Prediction — Cosine Similarity Before & After Discovery",
        fontsize=15, y=0.99,
    )

    for idx, row in enumerate(sample):
        ax   = axes[idx // 3][idx % 3]
        dots = [float(row[c]) for c in year_cols]
        disc = int(row["discovery_year"])

        ax.plot(years, dots, "o-", color="steelblue",
                markersize=4, linewidth=1.5, label="cos sim")
        ax.axvline(x=disc, color="crimson", linestyle="--", linewidth=2,
                   label=f"Discovered {disc}")

        ax.set_title(f"{row['codeA']}  \u2194  {row['codeB']}", fontsize=10)
        ax.legend(fontsize=8, loc="best")
        ax.set_xlabel("Year")
        ax.set_ylabel("Cosine similarity")
        ax.grid(True, alpha=0.25)
        ax.set_xticks(years[::2])

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = "output/innovation_plots.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure to {out_path}")

if __name__ == "__main__":
    main()
