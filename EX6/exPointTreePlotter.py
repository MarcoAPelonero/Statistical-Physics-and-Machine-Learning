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
bits = 10  # keep in sync with C++ code

# Aggregate exPointTwo across trials: group by train_size
grouped = df2.groupby('train_size').agg({
    'train_error_rate': ['mean', 'std'],
    'test_error_rate': ['mean', 'std']
}).reset_index()
# flatten columns
grouped.columns = ['train_size', 'train_err_mean', 'train_err_std', 'test_err_mean', 'test_err_std']

# compute alpha for grouped data
grouped['alpha'] = grouped['train_size'] / (2.0 * bits)
# sort by alpha
grouped = grouped.sort_values('alpha')

# exPointThree already contains alpha and epsilons
# ensure sorted by alpha
df3_sorted = df3.sort_values('alpha')

# Build the figure with two subplots and superimpose curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Subplot 1: Training error (empirical) vs alpha with theoretical training curve only
ax1.errorbar(grouped['alpha'], grouped['train_err_mean'],
             yerr=grouped['train_err_std'], marker='o', linestyle='-', capsize=4,
             label='Empirical training (mean ± std)', color='tab:blue')
ax1.plot(df3_sorted['alpha'], df3_sorted['epsilon_train'], '-', color='tab:green', lw=2, label='Theory: $\\epsilon_{train}$')
ax1.set_xlabel('Alpha (P / (2*bits))', fontsize=12)
ax1.set_ylabel('Training error', fontsize=12)
ax1.set_title('Training error: empirical vs theory', fontsize=13)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)

# Subplot 2: Generalization (test) error (empirical) vs alpha with theoretical generalization curve only
ax2.errorbar(grouped['alpha'], grouped['test_err_mean'],
             yerr=grouped['test_err_std'], marker='s', linestyle='-', capsize=4,
             label='Empirical test (mean ± std)', color='tab:red')
ax2.plot(df3_sorted['alpha'], df3_sorted['epsilon_theory'], '-', color='tab:orange', lw=2, label='Theory: $\\epsilon_{theory}$')
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
