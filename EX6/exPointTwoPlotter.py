import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Problem size (keep consistent with simulation)
bits = 30
N = 2 * bits  # feature dimension

# Optional: restrict maxpoint search to these alphas (P/N). If empty, no marker.
candidate_alphas = []  # e.g., [0.5, 1.0, 1.5, 2.0, 3.5, 7.0, 10.0, 20.0, 25.0, 30.0, 40.0, 50.0]

# If you want exact alpha mapping as used in the C++ simulation, set here:
alpha_values = []
alpha_values.extend(np.arange(0.5, 5.5, 0.5))
alpha_values.extend(np.arange(6.0, 16.0, 1.0))
alpha_values.extend(np.arange(20.0, 55.0, 5.0))
p_values = [int(a * N) for a in alpha_values]
alpha_map = dict(zip(p_values, alpha_values))

# Read the CSV data
data_path = Path('plots') / 'exPointTwo_results.csv'
df = pd.read_csv(data_path)

# Group by training size and compute statistics
grouped = df.groupby('train_size').agg({
    'train_error_rate': ['mean', 'std'],
    'test_error_rate': ['mean', 'std']
}).reset_index()

 # Flatten column names
grouped.columns = ['train_size', 'train_err_mean', 'train_err_std', 'test_err_mean', 'test_err_std']

# Map train_size (P) to load alpha = P/N using predefined mapping (avoids FP drift)
grouped['alpha'] = grouped['train_size'].map(alpha_map)
# Fallback: if some rows didn't map, compute directly
unmapped = grouped['alpha'].isna()
if unmapped.any():
    grouped.loc[unmapped, 'alpha'] = grouped.loc[unmapped, 'train_size'] / N

# Sort by alpha for nicer plots
grouped = grouped.sort_values('alpha')

# Number of trials used to compute the standard error (std / sqrt(N))
n_trials = 1000
# compute standard error columns
grouped['train_err_se'] = grouped['train_err_std'] / np.sqrt(n_trials)
grouped['test_err_se'] = grouped['test_err_std'] / np.sqrt(n_trials)
 # Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Training Error
ax1.errorbar(grouped['alpha'], grouped['train_err_mean'], 
             yerr=grouped['train_err_se'], 
             marker='o', linestyle='-', capsize=5, capthick=2,
             label='Training Error', color='blue', linewidth=2, markersize=6)
ax1.set_xlabel('Load (α = P/N)', fontsize=12)
ax1.set_ylabel('Error Rate', fontsize=12)
ax1.set_title('Training Error Rate vs Load', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=11)

# Plot 2: Generalization (Test) Error
ax2.errorbar(grouped['alpha'], grouped['test_err_mean'], 
             yerr=grouped['test_err_se'], 
             marker='s', linestyle='-', capsize=5, capthick=2,
             label='Generalization Error', color='red', linewidth=2, markersize=6)
ax2.set_xlabel('Load (α = P/N)', fontsize=12)
ax2.set_ylabel('Error Rate', fontsize=12)
ax2.set_title('Generalization Error Rate vs Load', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=11)

plt.tight_layout()

# Save the figure
output_path = Path('plots') / 'exPointTwo_analysis.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to {output_path}")

# Display the plot
plt.show()

# Print summary statistics
print("\n=== Summary Statistics ===")
print(grouped.to_string(index=False))

# --- Combined plot: training and test error on the same axes ---
fig2, ax_comb = plt.subplots(figsize=(8,6))

ax_comb.errorbar(grouped['alpha'], grouped['train_err_mean'], 
                 yerr=grouped['train_err_se'], marker='o', linestyle='-',
                 capsize=4, label='Training Error', color='blue')

ax_comb.errorbar(grouped['alpha'], grouped['test_err_mean'],
                 yerr=grouped['test_err_se'], marker='s', linestyle='-',
                 capsize=4, label='Generalization Error', color='red')

ax_comb.set_xlabel('Load (α = P/N)', fontsize=12)
ax_comb.set_ylabel('Error Rate', fontsize=12)
ax_comb.set_title('Training vs Generalization Error vs Load (means ± se)', fontsize=14)
ax_comb.grid(True, alpha=0.3)
ax_comb.legend(fontsize=11)

max_row = grouped.loc[grouped['train_err_mean'].idxmax()]
max_x, max_y = float(max_row['alpha']), float(max_row['train_err_mean'])

# Mark it
ax1.plot(
    max_x, max_y,
    marker='x',
    color='red',
    markersize=14,
    markeredgewidth=2.5,
    zorder=999   # <-- forces it to the very top
)
plt.tight_layout()
combined_path = Path('plots') / 'exPointTwo_combined.png'
plt.savefig(combined_path, dpi=300, bbox_inches='tight')
print(f"Combined plot saved to {combined_path}")
plt.show()
