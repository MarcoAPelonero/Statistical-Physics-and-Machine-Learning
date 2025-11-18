import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Files
path_two = Path('plots') / 'exPointTwo_results.csv'
path_three = Path('plots') / 'exPointThree_results.csv'

if not path_two.exists():
    raise FileNotFoundError(f"Missing file: {path_two}. Run exPointTwo generation first.")
if not path_three.exists():
    raise FileNotFoundError(f"Missing file: {path_three}. Run exPointThree generation first.")

# Read data
df2 = pd.read_csv(path_two)
df3 = pd.read_csv(path_three)

# Parameters (must match how exPointThree computed alpha)
bits = 30  # keep in sync with C++ code (N = 2 * bits = 60)
N = 2 * bits

# Define the exact alpha values used in the C++ simulation
alpha_values = []
alpha_values.extend(np.arange(0.5, 5.5, 0.5))
alpha_values.extend(np.arange(6.0, 16.0, 1.0))
alpha_values.extend(np.arange(20.0, 55.0, 5.0))

# Create a mapping from train_size back to the original alpha
# This avoids floating point inaccuracies from recalculating
p_values = [int(a * N) for a in alpha_values]
alpha_map = dict(zip(p_values, alpha_values))

# Aggregate exPointTwo across trials: group by train_size
grouped = df2.groupby('train_size').agg({
    'train_error_rate': ['mean', 'std'],
    'test_error_rate': ['mean', 'std']
}).reset_index()
# flatten columns
grouped.columns = ['train_size', 'train_err_mean', 'train_err_std', 'test_err_mean', 'test_err_std']

# Map train_size to the original alpha values
grouped['alpha'] = grouped['train_size'].map(alpha_map)
# remove any rows that didn't map (if any)
grouped.dropna(subset=['alpha'], inplace=True)
# sort by alpha
grouped = grouped.sort_values('alpha')

# Number of trials used to compute the standard error (std / sqrt(N))
n_trials = 1000
# compute standard error columns
grouped['train_err_se'] = grouped['train_err_std'] / np.sqrt(n_trials)
grouped['test_err_se'] = grouped['test_err_std'] / np.sqrt(n_trials)

# exPointThree already contains alpha and epsilons
# ensure sorted by alpha
df3_sorted = df3.sort_values('alpha')

# Build the figure with two subplots and superimpose curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Subplot 1: Training error (empirical) vs alpha with theoretical training curve only
ax1.errorbar(grouped['alpha'], grouped['train_err_mean'],
             yerr=grouped['train_err_se'], marker='o', linestyle='-', capsize=4,
             label='Empirical training (mean ± se)', color='tab:blue')
ax1.plot(df3_sorted['alpha'], df3_sorted['epsilon_train'], '-', color='tab:green', lw=2, label='Theory (Gaussian): $\\epsilon_{train}$')
if 'epsilon_mc_train' in df3_sorted.columns:
    ax1.plot(df3_sorted['alpha'], df3_sorted['epsilon_mc_train'], '--', color='tab:purple', lw=2, label='Monte Carlo (binary): $\\epsilon^{MC}_{train}$')
ax1.set_xlabel('Alpha (P / (2*bits))', fontsize=12)
ax1.set_ylabel('Training error', fontsize=12)
ax1.set_title('Training error: empirical vs theory', fontsize=13)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)

# Subplot 2: Generalization (test) error (empirical) vs alpha with theoretical generalization curve only
ax2.errorbar(grouped['alpha'], grouped['test_err_mean'],
             yerr=grouped['test_err_se'], marker='s', linestyle='-', capsize=4,
             label='Empirical test (mean ± se)', color='tab:red')
ax2.plot(df3_sorted['alpha'], df3_sorted['epsilon_theory'], '-', color='tab:orange', lw=2, label='Theory (Gaussian): $\\epsilon_{theory}$')
if 'epsilon_mc_test' in df3_sorted.columns:
    ax2.plot(df3_sorted['alpha'], df3_sorted['epsilon_mc_test'], '--', color='tab:purple', lw=2, label='Monte Carlo (binary): $\\epsilon^{MC}_{test}$')
ax2.set_xlabel('Alpha (P / (2*bits))', fontsize=12)
ax2.set_ylabel('Generalization error', fontsize=12)
ax2.set_title('Generalization error: empirical vs theory', fontsize=13)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)

plt.tight_layout()

# Save
out_path = Path('plots') / 'exPoint2_3_comparison.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Combined comparison plot saved to {out_path}")

# Show interactively
plt.show()

# Print brief summary
print('\nSummary (grouped empirical means):')
print(grouped[['train_size','alpha','train_err_mean','train_err_std','test_err_mean','test_err_std']].to_string(index=False))
