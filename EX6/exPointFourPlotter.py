import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Files - use experimental data from exPointTwo and theory from exPointFour
path_two = Path('plots') / 'exPointTwo_results.csv'
path_four = Path('plots') / 'exPointFour_results.csv'

if not path_two.exists():
    raise FileNotFoundError(f"Missing file: {path_two}. Run exPointTwo generation first.")
if not path_four.exists():
    raise FileNotFoundError(f"Missing file: {path_four}. Run exPointFour generation first.")

# Read experimental data
df2 = pd.read_csv(path_two)

# Read theoretical data
df_theory = pd.read_csv(path_four)

# Parameters
bits = 30
N = 2 * bits

# Define the exact alpha values used in the C++ simulation
alpha_values = []
alpha_values.extend(np.arange(0.5, 5.5, 0.5))
alpha_values.extend(np.arange(6.0, 16.0, 1.0))
alpha_values.extend(np.arange(20.0, 75.0, 5.0))

# Create mapping from train_size to alpha
p_values = [int(a * N) for a in alpha_values]
alpha_map = dict(zip(p_values, alpha_values))

# Aggregate experimental data: group by train_size and calculate mean errors
grouped = df2.groupby('train_size').agg({
    'train_error_rate': ['mean', 'std'],
    'test_error_rate': ['mean', 'std']
}).reset_index()
grouped.columns = ['train_size', 'train_err_mean', 'train_err_std', 'test_err_mean', 'test_err_std']

# Map train_size to alpha
grouped['alpha'] = grouped['train_size'].map(alpha_map)
grouped.dropna(subset=['alpha'], inplace=True)

# Calculate experimental generalization gap
grouped['gap_experimental'] = grouped['test_err_mean'] - grouped['train_err_mean']

# Number of trials for standard error
n_trials = 1000
grouped['gap_experimental_se'] = np.sqrt(grouped['train_err_std']**2 + grouped['test_err_std']**2) / np.sqrt(n_trials)

# Focus on alpha range 30-70
df_high_alpha_exp = grouped[(grouped['alpha'] >= 30) & (grouped['alpha'] <= 70)].copy()
df_high_alpha_theory = df_theory[(df_theory['alpha'] >= 30) & (df_theory['alpha'] <= 70)].copy()

# Calculate theoretical generalization gaps
df_high_alpha_theory['gap_gaussian_theory'] = df_high_alpha_theory['epsilon_theory'] - df_high_alpha_theory['epsilon_train']
df_high_alpha_theory['gap_mc'] = df_high_alpha_theory['epsilon_mc_test'] - df_high_alpha_theory['epsilon_mc_train']

print(f"Experimental data points (alpha 30-70): {len(df_high_alpha_exp)}")
print(f"Theoretical data points (alpha 30-70): {len(df_high_alpha_theory)}")
print(f"Alpha range: {df_high_alpha_exp['alpha'].min():.2f} to {df_high_alpha_exp['alpha'].max():.2f}")

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Plot EXPERIMENTAL generalization gap with error bars
ax.errorbar(df_high_alpha_exp['alpha'], df_high_alpha_exp['gap_experimental'],
            yerr=df_high_alpha_exp['gap_experimental_se'], marker='o', linestyle='-', capsize=4,
            label='Experimental: Test Error - Training Error', color='tab:blue', lw=2, markersize=7)

# Plot theoretical generalization gaps
ax.plot(df_high_alpha_theory['alpha'], df_high_alpha_theory['gap_gaussian_theory'], 
        '--', color='tab:green', lw=2,
        label='Gaussian Theory: $\\epsilon_{theory} - \\epsilon_{train}$')

ax.plot(df_high_alpha_theory['alpha'], df_high_alpha_theory['gap_mc'], 
        '--', color='tab:purple', lw=2,
        label='Monte Carlo (binary): $\\epsilon^{MC}_{test} - \\epsilon^{MC}_{train}$')

ax.set_xlabel('Alpha (P / N)', fontsize=13)
ax.set_ylabel('Generalization Gap (Test Error - Training Error)', fontsize=13)
ax.set_title('Generalization Gap vs Alpha (α = 30-70)', fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11, loc='best')

# Add horizontal line at zero for reference
ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

plt.tight_layout()

# Save
out_path = Path('plots') / 'exPointFour_generalization_gap.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"\nGeneralization gap plot saved to {out_path}")

# Show interactively
plt.show()

# Print summary statistics
print('\nSummary Statistics for Generalization Gap (Alpha 30-70):')
print(f"Experimental Gap - Mean: {df_high_alpha_exp['gap_experimental'].mean():.4f}, Std: {df_high_alpha_exp['gap_experimental'].std():.4f}")
print(f"Gaussian Theory Gap - Mean: {df_high_alpha_theory['gap_gaussian_theory'].mean():.4f}, Std: {df_high_alpha_theory['gap_gaussian_theory'].std():.4f}")
print(f"Monte Carlo Gap - Mean: {df_high_alpha_theory['gap_mc'].mean():.4f}, Std: {df_high_alpha_theory['gap_mc'].std():.4f}")
print('\nDetailed experimental values:')
print(df_high_alpha_exp[['alpha', 'train_err_mean', 'test_err_mean', 'gap_experimental']].to_string(index=False))
