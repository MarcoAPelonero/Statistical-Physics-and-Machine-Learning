#!/usr/bin/env python3
"""
Read culture distribution files `culture_distribution_fdim_*.txt` produced by
`tests/test_clustersize.cpp` and plot:
 - Overlapped cluster-size distributions (linear-linear)
 - Overlapped cluster-size distributions (log-log)
 - Biggest culture size (fraction of nodes) vs feature dimension

Usage:
  python scripts/plot_culture_distributions.py --data-dir data/culture_data --out-dir figures

If matplotlib is not installed: `python -m pip install matplotlib numpy`
"""
import argparse
import glob
import os
import re
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt


def read_distribution_file(path):
    """Return list of cluster sizes (ints) read from a file."""
    with open(path, 'r') as f:
        line = f.readline().strip()
    if not line:
        return []
    parts = line.split()
    try:
        counts = [int(p) for p in parts]
    except ValueError:
        # try to filter non-int tokens
        counts = []
        for p in parts:
            try:
                counts.append(int(p))
            except Exception:
                continue
    return counts


def parse_fdim_from_name(name):
    m = re.search(r'fdim_(\d+)', name)
    if m:
        return int(m.group(1))
    # fallback: try any integer in basename
    m2 = re.search(r'(\d+)', os.path.basename(name))
    return int(m2.group(1)) if m2 else None


def build_size_distribution(counts):
    """Given a list of cluster sizes (one entry per cluster), return arrays (sizes, n_s)
    where n_s is the number of clusters of size s.
    """
    c = Counter(counts)
    if not c:
        return np.array([]), np.array([])
    sizes = np.array(sorted(c.keys()), dtype=int)
    ns = np.array([c[s] for s in sizes], dtype=float)
    return sizes, ns


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='data/culture_data', help='Directory with distribution files')
    p.add_argument('--pattern', default='culture_distribution_fdim_*.txt', help='Filename pattern')
    p.add_argument('--out-dir', default='figures', help='Output directory for PNGs')
    p.add_argument('--show', action='store_true', help='Show figures interactively')
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, args.pattern)))
    if not files:
        print('No files found in', args.data_dir)
        return

    os.makedirs(args.out_dir, exist_ok=True)

    all_data = []  # list of dicts: {fdim, sizes, ns, ns_norm, max_cluster, total_nodes}

    for fp in files:
        fdim = parse_fdim_from_name(fp)
        counts = read_distribution_file(fp)
        if not counts:
            print('Warning: empty or unreadable file', fp)
            continue
        total_nodes = sum(counts)
        if total_nodes == 0:
            print('Warning: total nodes zero in', fp)
            continue
        sizes, ns = build_size_distribution(counts)
        ns_norm = ns / ns.sum()  # normalize by number of clusters
        max_cluster = max(counts)
        all_data.append({'fdim': fdim, 'sizes': sizes, 'ns': ns, 'ns_norm': ns_norm, 'max_cluster': max_cluster, 'total_nodes': total_nodes})

    # sort by fdim
    all_data.sort(key=lambda x: x['fdim'])

    # Plot linear-linear overlap
    plt.figure(figsize=(10, 6))
    cmap = plt.get_cmap('tab10')
    for i, d in enumerate(all_data):
        plt.plot(d['sizes'], d['ns_norm'], label=f"f={d['fdim']}", marker='o', linestyle='-')
    plt.xlabel('Cluster size (s)')
    plt.ylabel('P(s) = fraction of clusters')
    plt.title('Cluster size distributions (linear)')
    plt.legend(loc='best', ncol=2)
    plt.grid(True, linestyle='--', alpha=0.4)
    out_lin = os.path.join(args.out_dir, 'culture_distributions_linear.png')
    plt.tight_layout()
    plt.savefig(out_lin, dpi=200)
    print('Saved', out_lin)

    # Plot log-log overlap and fit power-law slopes
    plt.figure(figsize=(10, 6))
    for i, d in enumerate(all_data):
        sizes = d['sizes']
        ns_norm = d['ns_norm']
        # avoid zeros in sizes and probabilities
        mask = (sizes > 0) & (ns_norm > 0)
        if mask.sum() == 0:
            continue
        # plot empirical points
        plt.loglog(sizes[mask], ns_norm[mask], label=f"f={d['fdim']}", marker='o', linestyle='none', color=cmap(i))
        # fit a power law P(s) ~ s^a  => log P = a log s + b
        if mask.sum() >= 2:
            logx = np.log(sizes[mask].astype(float))
            logy = np.log(ns_norm[mask].astype(float))
            a, b = np.polyfit(logx, logy, 1)
            # build smooth fit line over the observed size range
            x_fit = np.linspace(sizes[mask].min(), sizes[mask].max(), 200)
            y_fit = np.exp(b) * (x_fit ** a)
            plt.loglog(x_fit, y_fit, linestyle='--', linewidth=1.5, color=cmap(i),
                       label=f"fit f={d['fdim']} (slope={a:.2f})")
            print(f"f={d['fdim']}: fit slope (log-log) = {a:.4f}")
    plt.xlabel('Cluster size (s)')
    plt.ylabel('P(s)')
    plt.title('Cluster size distributions (log-log) with power-law fits')
    plt.legend(loc='best', ncol=2)
    plt.grid(True, which='both', linestyle='--', alpha=0.4)
    out_log = os.path.join(args.out_dir, 'culture_distributions_loglog.png')
    plt.tight_layout()
    plt.savefig(out_log, dpi=200)
    print('Saved', out_log)

    # Plot biggest cluster fraction vs fdim
    fds = [d['fdim'] for d in all_data]
    L_fracs = [d['max_cluster'] / d['total_nodes'] for d in all_data]
    plt.figure(figsize=(8, 5))
    plt.plot(fds, L_fracs, marker='o', linestyle='-')
    plt.xlabel('Feature dimension (f_dim)')
    plt.ylabel('Largest culture size (fraction of nodes)')
    plt.title('Largest culture fraction vs feature dimension')
    plt.grid(True, linestyle='--', alpha=0.4)
    out_big = os.path.join(args.out_dir, 'biggest_cluster_vs_fdim.png')
    plt.tight_layout()
    plt.savefig(out_big, dpi=200)
    print('Saved', out_big)

    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
