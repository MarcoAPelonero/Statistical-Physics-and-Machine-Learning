import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple


def import_data_from_file(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load avalanche data from a CSV or whitespace-delimited file.

    The function tries to read two columns (size, duration). It first
    attempts a comma-delimited load (CSV). If that fails it falls back to
    whitespace-delimited parsing. Returns two numpy arrays (sizes, durations).
    """
    sizes = []
    durations = []

    # Read line-by-line and extract the first two numeric values per line.
    # This is robust to comment/header lines or explanatory text in the file.
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                continue
            # Normalize commas to whitespace so we can split consistently
            parts = [p for p in line.replace(',', ' ').split() if p]
            if len(parts) < 2:
                # try next line if not enough tokens
                continue

            vals = []
            for token in parts:
                try:
                    v = float(token)
                except ValueError:
                    # skip non-numeric tokens (e.g., textual headers)
                    continue
                vals.append(v)
                if len(vals) == 2:
                    break

            if len(vals) == 2:
                sizes.append(vals[0])
                durations.append(vals[1])
            else:
                # couldn't find two numeric values on this line; skip it
                continue

    return np.array(sizes, dtype=float), np.array(durations, dtype=float)


def _hist_on_ax(ax, data, bins='auto', title=None, xlabel=None, ylabel=None, log_scale=True, color='C0'):
    """Helper to draw histogram on given Axes. Handles log spacing for bins if log_scale=True."""
    data = np.asarray(data)
    data = data[np.isfinite(data)]
    if data.size == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        return

    if log_scale:
        positives = data[data > 0]
        if positives.size >= 2:
            minv = positives.min()
            maxv = positives.max()
            # avoid degenerate bins
            if minv <= 0:
                minv = positives[positives > 0].min()
            num_bins = 50
            bins = np.logspace(np.log10(minv), np.log10(maxv), num=num_bins)
        else:
            bins = 'auto'

    counts, bin_edges = np.histogram(data, bins=bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    ax.bar(bin_centers, counts, width=np.diff(bin_edges), edgecolor='black', align='center', color=color)
    if log_scale:
        ax.set_xscale('log')
        # For y, avoid zeros when using logscale by setting nonzero bottom
        ax.set_yscale('log')

    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    # ax.grid(True, which='both', ls='--')


def plot_ccdf(ax, data, title='CCDF', xlabel='Value', ylabel='CCDF', log_scale=True, color='C3'):
    """Plot the complementary cumulative distribution function (CCDF) of data on ax."""
    data = np.asarray(data)
    data = data[np.isfinite(data)]
    data = data[data > 0]
    if data.size == 0:
        ax.text(0.5, 0.5, 'No positive data', ha='center', va='center')
        return

    sorted_data = np.sort(data)
    n = sorted_data.size
    # For each sorted value x_i, CCDF = fraction >= x_i
    ccdf = 1.0 - np.arange(0, n) / n

    ax.step(sorted_data, ccdf, where='post', color=color)
    if log_scale:
        ax.set_xscale('log')
        ax.set_yscale('log')

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which='both', ls='--')


def plot_scatter(ax, x, y, title=None, xlabel=None, ylabel=None, log_scale=True, color='C2'):
    ax.scatter(x, y, alpha=0.6, color=color)
    if log_scale:
        # guard against non-positive values for log scale
        if np.any(x <= 0) or np.any(y <= 0):
            # use symmetric log if there are zeros/negatives? For now, fallback to linear
            ax.set_xscale('linear')
            ax.set_yscale('linear')
        else:
            ax.set_xscale('log')
            ax.set_yscale('log')
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, which='both', ls='--')


def plot_all(sizes, durations, log_scale=True, figsize=(12, 10)):
    """Create a 2x2 subplot layout with size dist, duration dist, scatter, and CCDF."""
    sizes = np.asarray(sizes)
    durations = np.asarray(durations)

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Top-left: size distribution
    _hist_on_ax(axes[0, 0], sizes, title='Avalanche Size Distribution', xlabel='Avalanche Size', ylabel='Frequency', log_scale=log_scale, color='C0')

    # Top-right: duration distribution
    _hist_on_ax(axes[0, 1], durations, title='Avalanche Duration Distribution', xlabel='Avalanche Duration', ylabel='Frequency', log_scale=log_scale, color='C1')

    # Bottom-left: size vs duration scatter
    plot_scatter(axes[1, 0], durations, sizes, title='Avalanche Size vs Duration', xlabel='Duration', ylabel='Size', log_scale=log_scale, color='C2')

    # Bottom-right: CCDF of sizes (additional informative plot)
    plot_ccdf(axes[1, 1], sizes, title='Size CCDF', xlabel='Avalanche Size', ylabel='P(X >= x)', log_scale=log_scale, color='C3')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    sizes, durations = import_data_from_file('avalanches.csv')
    # Quick sanity: ensure arrays
    sizes = np.asarray(sizes)
    durations = np.asarray(durations)
    if sizes.size == 0 or durations.size == 0:
        print('No data found in avalanches.csv')
    else:
        plot_all(sizes, durations)
