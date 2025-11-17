import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

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

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Training Error
ax1.errorbar(grouped['train_size'], grouped['train_err_mean'], 
             yerr=grouped['train_err_std'], 
             marker='o', linestyle='-', capsize=5, capthick=2,
             label='Training Error', color='blue', linewidth=2, markersize=6)
ax1.set_xlabel('Training Set Size (P)', fontsize=12)
ax1.set_ylabel('Error Rate', fontsize=12)
ax1.set_title('Training Error Rate vs Training Set Size', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=11)

# Plot 2: Generalization (Test) Error
ax2.errorbar(grouped['train_size'], grouped['test_err_mean'], 
             yerr=grouped['test_err_std'], 
             marker='s', linestyle='-', capsize=5, capthick=2,
             label='Generalization Error', color='red', linewidth=2, markersize=6)
ax2.set_xlabel('Training Set Size (P)', fontsize=12)
ax2.set_ylabel('Error Rate', fontsize=12)
ax2.set_title('Generalization Error Rate vs Training Set Size', fontsize=14, fontweight='bold')
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

ax_comb.errorbar(grouped['train_size'], grouped['train_err_mean'],
                 yerr=grouped['train_err_std'], marker='o', linestyle='-',
                 capsize=4, label='Training Error', color='blue')

ax_comb.errorbar(grouped['train_size'], grouped['test_err_mean'],
                 yerr=grouped['test_err_std'], marker='s', linestyle='-',
                 capsize=4, label='Generalization Error', color='red')

ax_comb.set_xlabel('Training Set Size (P)', fontsize=12)
ax_comb.set_ylabel('Error Rate', fontsize=12)
ax_comb.set_title('Training vs Generalization Error (means ± std)', fontsize=14)
ax_comb.grid(True, alpha=0.3)
ax_comb.legend(fontsize=11)

plt.tight_layout()
combined_path = Path('plots') / 'exPointTwo_combined.png'
plt.savefig(combined_path, dpi=300, bbox_inches='tight')
print(f"Combined plot saved to {combined_path}")
plt.show()
