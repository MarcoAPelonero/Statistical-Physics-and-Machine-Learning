import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load data (use Path for cross-platform clarity)
def exPointTwoPlotter():
    data_path = Path('data') / 'exPointTwo_results.csv'
    data = np.loadtxt(data_path, delimiter=',', skiprows=1)

    alpha = data[:, 0]
    train_error_rate = data[:, 1]
    test_error_rate = data[:, 2]

    # Aggregate by unique alpha values
    unique_alpha = np.unique(alpha)
    mean_train_error = []
    mean_test_error = []
    se_train_error = []
    se_test_error = []

    for a in unique_alpha:
        mask = alpha == a
        n = np.sum(mask)
        train_vals = train_error_rate[mask]
        test_vals = test_error_rate[mask]
        mean_train_error.append(np.mean(train_vals))
        mean_test_error.append(np.mean(test_vals))
        # Use sample standard deviation (ddof=1) when possible; fallback to 0 for single sample
        se_train_error.append(np.std(train_vals, ddof=1) / np.sqrt(n) if n > 1 else 0.0)
        se_test_error.append(np.std(test_vals, ddof=1) / np.sqrt(n) if n > 1 else 0.0)

    mean_train_error = np.array(mean_train_error)
    mean_test_error = np.array(mean_test_error)
    se_train_error = np.array(se_train_error)
    se_test_error = np.array(se_test_error)

    # Seaborn theming for nicer default aesthetics
    sns.set_theme(style='whitegrid', palette='tab10')
    plt.figure(figsize=(9, 6), dpi=100)

    # Plot training error with shaded standard-error band
    plt.plot(unique_alpha, mean_train_error, marker='o', linewidth=2.2, label='Training Error Rate')
    plt.fill_between(unique_alpha, mean_train_error - se_train_error, mean_train_error + se_train_error, alpha=0.25)

    # Plot test error with shaded standard-error band
    plt.plot(unique_alpha, mean_test_error, marker='s', linewidth=2.2, label='Test Error Rate')
    plt.fill_between(unique_alpha, mean_test_error - se_test_error, mean_test_error + se_test_error, alpha=0.25)

    plt.xlabel('Alpha (P/N)')
    plt.ylabel('Error Rate')
    plt.title('Training and Test Error Rates vs Alpha')
    plt.legend(frameon=True)
    plt.grid(linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def extraPointOnePlotter():
    data_path = Path('data') / 'extraPointOne_results.csv'
    data = np.loadtxt(data_path, delimiter=',', skiprows=1)

    alpha = data[:, 0]
    train_error_rate = data[:, 1]
    test_error_rate = data[:, 2]

    # Aggregate by unique alpha values
    unique_alpha = np.unique(alpha)
    mean_train_error = []
    mean_test_error = []
    se_train_error = []
    se_test_error = []

    for a in unique_alpha:
        mask = alpha == a
        n = np.sum(mask)
        train_vals = train_error_rate[mask]
        test_vals = test_error_rate[mask]
        mean_train_error.append(np.mean(train_vals))
        mean_test_error.append(np.mean(test_vals))
        # Use sample standard deviation (ddof=1) when possible; fallback to 0 for single sample
        se_train_error.append(np.std(train_vals, ddof=1) / np.sqrt(n) if n > 1 else 0.0)
        se_test_error.append(np.std(test_vals, ddof=1) / np.sqrt(n) if n > 1 else 0.0)

    mean_train_error = np.array(mean_train_error)
    mean_test_error = np.array(mean_test_error)
    se_train_error = np.array(se_train_error)
    se_test_error = np.array(se_test_error)

    # Seaborn theming for nicer default aesthetics
    sns.set_theme(style='whitegrid', palette='tab10')
    plt.figure(figsize=(9, 6), dpi=100)

    # Plot training error with shaded standard-error band
    plt.plot(unique_alpha, mean_train_error, marker='o', linewidth=2.2, label='Training Error Rate')
    plt.fill_between(unique_alpha, mean_train_error - se_train_error, mean_train_error + se_train_error, alpha=0.25)

    # Plot test error with shaded standard-error band
    plt.plot(unique_alpha, mean_test_error, marker='s', linewidth=2.2, label='Test Error Rate')
    plt.fill_between(unique_alpha, mean_test_error - se_test_error, mean_test_error + se_test_error, alpha=0.25)

    plt.xlabel('Alpha (P/N)')
    plt.ylabel('Error Rate')
    plt.title('Training and Test Error Rates vs Alpha (Ridge Regression)')
    plt.legend(frameon=True)
    plt.grid(linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def extraPointFivePlotter():
    data_path = Path('data') / 'exPointFive_results.csv'
    data = np.loadtxt(data_path, delimiter=',', skiprows=1)

    alpha = data[:, 0]
    train_error_rate = data[:, 1]
    test_error_rate = data[:, 2]

    # Aggregate by unique alpha values
    unique_alpha = np.unique(alpha)
    mean_train_error = []
    mean_test_error = []
    se_train_error = []
    se_test_error = []

    for a in unique_alpha:
        mask = alpha == a
        n = np.sum(mask)
        train_vals = train_error_rate[mask]
        test_vals = test_error_rate[mask]
        mean_train_error.append(np.mean(train_vals))
        mean_test_error.append(np.mean(test_vals))
        # Use sample standard deviation (ddof=1) when possible; fallback to 0 for single sample
        se_train_error.append(np.std(train_vals, ddof=1) / np.sqrt(n) if n > 1 else 0.0)
        se_test_error.append(np.std(test_vals, ddof=1) / np.sqrt(n) if n > 1 else 0.0)

    mean_train_error = np.array(mean_train_error)
    mean_test_error = np.array(mean_test_error)
    se_train_error = np.array(se_train_error)
    se_test_error = np.array(se_test_error)

    # Seaborn theming for nicer default aesthetics
    sns.set_theme(style='whitegrid', palette='tab10')
    plt.figure(figsize=(9, 6), dpi=100)

    # Plot training error with shaded standard-error band
    plt.plot(unique_alpha, mean_train_error, marker='o', linewidth=2.2, label='Training Error Rate')
    plt.fill_between(unique_alpha, mean_train_error - se_train_error, mean_train_error + se_train_error, alpha=0.25)

    # Plot test error with shaded standard-error band
    plt.plot(unique_alpha, mean_test_error, marker='s', linewidth=2.2, label='Test Error Rate')
    plt.fill_between(unique_alpha, mean_test_error - se_test_error, mean_test_error + se_test_error, alpha=0.25)

    plt.xlabel('Alpha (P/N)')
    plt.ylabel('Error Rate')
    plt.title('Training and Test Error Rates vs Alpha (Adaline)')
    plt.legend(frameon=True)
    plt.grid(linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def extraPointSixPlotter():
    # Load both datasets
    data_path_two = Path('data') / 'extraPointOne_results.csv'
    data_path_five = Path('data') / 'exPointFive_results.csv'
    data_two = np.loadtxt(data_path_two, delimiter=',', skiprows=1)
    data_five = np.loadtxt(data_path_five, delimiter=',', skiprows=1)

    # Extract data from exPointTwo (Pseudoinverse)
    alpha_two = data_two[:, 0]
    train_error_two = data_two[:, 1]
    test_error_two = data_two[:, 2]

    # Extract data from exPointFive (Adaline)
    alpha_five = data_five[:, 0]
    train_error_five = data_five[:, 1]
    test_error_five = data_five[:, 2]

    # Aggregate exPointTwo by unique alpha values
    unique_alpha_two = np.unique(alpha_two)
    mean_train_two = []
    mean_test_two = []
    se_train_two = []
    se_test_two = []

    for a in unique_alpha_two:
        mask = alpha_two == a
        n = np.sum(mask)
        train_vals = train_error_two[mask]
        test_vals = test_error_two[mask]
        mean_train_two.append(np.mean(train_vals))
        mean_test_two.append(np.mean(test_vals))
        se_train_two.append(np.std(train_vals, ddof=1) / np.sqrt(n) if n > 1 else 0.0)
        se_test_two.append(np.std(test_vals, ddof=1) / np.sqrt(n) if n > 1 else 0.0)

    mean_train_two = np.array(mean_train_two)
    mean_test_two = np.array(mean_test_two)
    se_train_two = np.array(se_train_two)
    se_test_two = np.array(se_test_two)

    # Aggregate exPointFive by unique alpha values
    unique_alpha_five = np.unique(alpha_five)
    mean_train_five = []
    mean_test_five = []
    se_train_five = []
    se_test_five = []

    for a in unique_alpha_five:
        mask = alpha_five == a
        n = np.sum(mask)
        train_vals = train_error_five[mask]
        test_vals = test_error_five[mask]
        mean_train_five.append(np.mean(train_vals))
        mean_test_five.append(np.mean(test_vals))
        se_train_five.append(np.std(train_vals, ddof=1) / np.sqrt(n) if n > 1 else 0.0)
        se_test_five.append(np.std(test_vals, ddof=1) / np.sqrt(n) if n > 1 else 0.0)

    mean_train_five = np.array(mean_train_five)
    mean_test_five = np.array(mean_test_five)
    se_train_five = np.array(se_train_five)
    se_test_five = np.array(se_test_five)

    # Create superimposed plot
    sns.set_theme(style='whitegrid', palette='tab10')
    plt.figure(figsize=(12, 7), dpi=100)

    # Plot exPointTwo (Pseudoinverse) - Test Error
    plt.plot(unique_alpha_two, mean_test_two, marker='s', linewidth=2.2, label='Test Error (Pseudoinverse)', color='C0')
    plt.fill_between(unique_alpha_two, mean_test_two - se_test_two, mean_test_two + se_test_two, alpha=0.15, color='C0')

    # Plot exPointFive (Adaline) - Test Error
    plt.plot(unique_alpha_five, mean_test_five, marker='^', linewidth=2.2, label='Test Error (Adaline)', color='C1')
    plt.fill_between(unique_alpha_five, mean_test_five - se_test_five, mean_test_five + se_test_five, alpha=0.15, color='C1')

    # Plot exPointTwo (Pseudoinverse) - Training Error
    plt.plot(unique_alpha_two, mean_train_two, marker='o', linewidth=2.2, label='Training Error (Pseudoinverse)', linestyle='--', color='C0')
    plt.fill_between(unique_alpha_two, mean_train_two - se_train_two, mean_train_two + se_train_two, alpha=0.15, color='C0')

    # Plot exPointFive (Adaline) - Training Error
    plt.plot(unique_alpha_five, mean_train_five, marker='D', linewidth=2.2, label='Training Error (Adaline)', linestyle='--', color='C1')
    plt.fill_between(unique_alpha_five, mean_train_five - se_train_five, mean_train_five + se_train_five, alpha=0.15, color='C1')

    plt.xlabel('Alpha (P/N)')
    plt.ylabel('Error Rate')
    plt.title('Comparison: Pseudoinverse vs Adaline')
    plt.legend(frameon=True, fontsize=10)
    plt.grid(linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # exPointTwoPlotter()
    # extraPointOnePlotter()
    extraPointFivePlotter()
    extraPointSixPlotter()