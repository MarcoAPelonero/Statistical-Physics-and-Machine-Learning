import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Read results from ExPointFive (this will replace the old `dataset.csv` usage)
# ExPointFive uses bits=10 -> N = 2 * bits = 20
bits_five = 10
N_five = 2 * bits_five
expoint_five_path = Path('plots') / 'exPointFive_results.csv'
df_expoint_five = pd.read_csv(expoint_five_path)

# Group ExPointFive by training size and compute statistics on test error rate
grouped_five = df_expoint_five.groupby('train_size').agg({
    'test_error_rate': ['mean', 'std']
}).reset_index()
grouped_five.columns = ['train_size', 'test_err_mean', 'test_err_std']
grouped_five['alpha'] = grouped_five['train_size'] / N_five

# Read the exPointTwo results (kept for comparison)
expoint_two_path = Path('plots') / 'exPointTwo_results.csv'
df_expoint = pd.read_csv(expoint_two_path)

# Group exPointTwo by training size and compute statistics
grouped = df_expoint.groupby('train_size').agg({
    'test_error_rate': ['mean', 'std']
}).reset_index()

# Flatten column names
grouped.columns = ['train_size', 'test_err_mean', 'test_err_std']

# Compute alpha for exPointTwo (assuming N=60 from the original code)
N_expoint = 60
grouped['alpha'] = grouped['train_size'] / N_expoint

# Number of trials to compute standard error
n_trials_five = 100   # From exPointFive (see exPoints.cpp)
n_trials_expoint = 1000  # From exPointTwo

# Compute standard errors
grouped_five['test_err_se'] = grouped_five['test_err_std'] / np.sqrt(n_trials_five)
grouped['test_err_se'] = grouped['test_err_std'] / np.sqrt(n_trials_expoint)

# Create the comparison plot: ExPointFive vs ExPointTwo
fig, ax = plt.subplots(figsize=(10, 7))

# Plot ExPointFive results
ax.errorbar(grouped_five['alpha'], grouped_five['test_err_mean'], 
            yerr=grouped_five['test_err_se'],
            marker='o', linestyle='-', capsize=5, capthick=2,
            label=f'Noisy Perceptron', 
            color='blue', linewidth=2, markersize=7, alpha=0.8)

# Plot ExPointTwo results
ax.errorbar(grouped['alpha'], grouped['test_err_mean'],
            yerr=grouped['test_err_se'],
            marker='s', linestyle='--', capsize=5, capthick=2,
            label=f'Hebbian Learning', 
            color='red', linewidth=2, markersize=7, alpha=0.8)

ax.set_xlabel('Load (α = P/N)', fontsize=14, fontweight='bold')
ax.set_ylabel('Generalization Error Rate', fontsize=14, fontweight='bold')
ax.set_title('Generalization Error Comparison: Noisy Perceptron vs Hebbian Learning', 
             fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(fontsize=12, loc='best')

# Add some styling
ax.set_xlim(left=0)
ax.set_ylim(bottom=0, top=0.5)

plt.tight_layout()

# Save the figure
output_path = Path('plots') / 'exPointFive_comparison.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Comparison plot saved to {output_path}")

# Display the plot
plt.show()

# Print summary statistics
print("\n=== ExPointFive Summary ===")
print(grouped_five[['train_size', 'alpha', 'test_err_mean', 'test_err_std']].to_string(index=False))

print("\n=== ExPointTwo Summary (first 10 rows) ===")
print(grouped.head(10).to_string(index=False))

# Create a second plot showing just the ExPointFive curve
fig2, ax2 = plt.subplots(figsize=(9, 6))

ax2.errorbar(grouped_five['alpha'], grouped_five['test_err_mean'], 
             yerr=grouped_five['test_err_se'],
             marker='o', linestyle='-', capsize=5, capthick=2,
             label=f'Generalization Error (N={N_five})', 
             color='darkblue', linewidth=2.5, markersize=8)

ax2.fill_between(grouped_five['alpha'], 
                  grouped_five['test_err_mean'] - grouped_five['test_err_se'],
                  grouped_five['test_err_mean'] + grouped_five['test_err_se'],
                  alpha=0.2, color='blue')

ax2.set_xlabel('Load (α = P/N)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Generalization Error Rate', fontsize=14, fontweight='bold')
ax2.set_title(f'Generalization Error Curve (ExPointFive, N={N_five})', 
              fontsize=16, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.legend(fontsize=12, loc='best')
ax2.set_xlim(left=0)
ax2.set_ylim(bottom=0, top=0.45)

plt.tight_layout()

# Save the individual plot
individual_path = Path('plots') / 'exPointFive_dataset_only.png'
plt.savefig(individual_path, dpi=300, bbox_inches='tight')
print(f"Individual dataset plot saved to {individual_path}")

plt.show()

print("\n=== Analysis Complete ===")
print(f"ExPointFive: {len(grouped_five)} data points")
print(f"ExPointTwo: {len(grouped)} data points")
