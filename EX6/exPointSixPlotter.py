import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_and_group_train_test(path, N, n_trials=None):
    df = pd.read_csv(path)
    grouped = df.groupby('train_size').agg({
        'train_error_rate': ['mean', 'std'],
        'test_error_rate': ['mean', 'std']
    }).reset_index()
    grouped.columns = ['train_size', 'train_err_mean', 'train_err_std', 'test_err_mean', 'test_err_std']
    grouped['alpha'] = grouped['train_size'] / float(N)

    # Infer number of trials if not provided
    if n_trials is None:
        try:
            inferred = int(df['trial'].nunique())
            n_trials = inferred if inferred > 0 else 1
        except Exception:
            n_trials = 1

    grouped['train_err_se'] = grouped['train_err_std'] / np.sqrt(n_trials)
    grouped['test_err_se'] = grouped['test_err_std'] / np.sqrt(n_trials)
    grouped = grouped.sort_values('alpha')
    return grouped


def main():
    plots_dir = Path('plots')
    # Paths
    path_six = plots_dir / 'exPointSix_results.csv'
    path_five = plots_dir / 'exPointFive_results.csv'
    path_two = plots_dir / 'exPointTwo_results.csv'

    if not path_six.exists():
        raise FileNotFoundError(f"Missing file: {path_six}. Run exPointSix generation first.")

    # N values must match the C++ simulation: exPointFive & Six use bits=10 -> N=20; exPointTwo uses bits=30 -> N=60
    N_six = 2 * 10
    N_five = 2 * 10
    N_two = 2 * 30

    # number of trials used in C++ code
    n_trials_six = 100
    n_trials_five = 100
    n_trials_two = 1000

    # Load and group datasets (train + test)
    grouped_six = load_and_group_train_test(path_six, N_six, n_trials_six)

    grouped_five = None
    if path_five.exists():
        grouped_five = load_and_group_train_test(path_five, N_five, n_trials_five)

    grouped_two = None
    if path_two.exists():
        grouped_two = load_and_group_train_test(path_two, N_two, n_trials_two)

    # --- Plot 1: ExPointSix dataset with Training and Generalization error (two subplots) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Training error subplot
    ax1.errorbar(grouped_six['alpha'], grouped_six['train_err_mean'],
                 yerr=grouped_six['train_err_se'], marker='o', linestyle='-',
                 capsize=4, label='Training Error', color='C0')
    ax1.set_xlabel('Load (α = P/N)', fontsize=12)
    ax1.set_ylabel('Training Error Rate', fontsize=12)
    ax1.set_title('Training Error vs Load (ExPointSix)', fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Generalization error subplot
    ax2.errorbar(grouped_six['alpha'], grouped_six['test_err_mean'],
                 yerr=grouped_six['test_err_se'], marker='s', linestyle='-',
                 capsize=4, label='Generalization Error', color='C1')
    ax2.set_xlabel('Load (α = P/N)', fontsize=12)
    ax2.set_ylabel('Generalization Error Rate', fontsize=12)
    ax2.set_title('Generalization Error vs Load (ExPointSix)', fontsize=14)
    ax2.grid(True, alpha=0.3)

    ax1.set_xlim(left=0)
    ax2.set_xlim(left=0)
    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)

    out1 = plots_dir / 'exPointSix_analysis.png'
    plt.tight_layout()
    plt.savefig(out1, dpi=300, bbox_inches='tight')
    print(f"Saved ExPointSix train/test plot to {out1}")
    plt.show()

    # --- Plot 2: Comparison between exPointTwo, exPointFive and exPointSix (generalization error) ---
    fig2, ax2 = plt.subplots(figsize=(10, 7))

    ax2.errorbar(grouped_six['alpha'], grouped_six['test_err_mean'],
                 yerr=grouped_six['test_err_se'], marker='o', linestyle='-',
                 capsize=4, label='ExPointSix (Perceptron)', color='C0')

    if grouped_five is not None:
        ax2.errorbar(grouped_five['alpha'], grouped_five['test_err_mean'],
                     yerr=grouped_five['test_err_se'], marker='s', linestyle='--',
                     capsize=4, label='ExPointFive (Noisy Perceptron)', color='C1')

    if grouped_two is not None:
        ax2.errorbar(grouped_two['alpha'], grouped_two['test_err_mean'],
                     yerr=grouped_two['test_err_se'], marker='^', linestyle='-.',
                     capsize=4, label='ExPointTwo (Hebbian)', color='C2')

    ax2.set_xlabel('Load (α = P/N)', fontsize=12)
    ax2.set_ylabel('Generalization Error Rate', fontsize=12)
    ax2.set_title('Generalization Error Comparison: ExPointTwo / Five / Six', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11, loc='best')
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0)

    out2 = plots_dir / 'exPoint_2_5_6_comparison.png'
    plt.tight_layout()
    plt.savefig(out2, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to {out2}")
    plt.show()

    # Print brief summaries
    print('\n=== ExPointSix Summary ===')
    print(grouped_six.to_string(index=False))

    if grouped_five is not None:
        print('\n=== ExPointFive Summary ===')
        print(grouped_five.to_string(index=False))

    if grouped_two is not None:
        print('\n=== ExPointTwo Summary (partial) ===')
        print(grouped_two.head(15).to_string(index=False))


if __name__ == '__main__':
    main()
