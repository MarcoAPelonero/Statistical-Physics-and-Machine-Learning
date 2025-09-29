#!/usr/bin/env python3
"""
Plotting script for funcUtils test data.
Visualizes the generated data files from test_funcUtils.cpp
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Define data directory
DATA_DIR = "testFuncUtilsData"

def ensure_output_dir():
    """Create output directory if it doesn't exist"""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created directory: {DATA_DIR}")

def plot_hidden_functions():
    """Plot the hidden functions A and B"""
    data_file = os.path.join(DATA_DIR, 'hidden_functions.dat')
    if not os.path.exists(data_file):
        print(f"Warning: {data_file} not found. Run test_funcUtils first.")
        return
    
    data = np.loadtxt(data_file)
    x = data[:, 0]
    hidden_a = data[:, 1]
    hidden_b = data[:, 2]
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(x, hidden_a, 'b-', linewidth=2, label='hiddenFunctionA(x) = 2x')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Hidden Function A')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(x, hidden_b, 'r-', linewidth=2, label='hiddenFunctionB(x) = 2x - 10x⁵ + 15x¹⁰')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Hidden Function B')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    output_file = os.path.join(DATA_DIR, 'hidden_functions.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ Hidden functions plotted and saved as {output_file}")

def plot_noisy_data():
    """Plot noisy data with different noise levels"""
    data_file = os.path.join(DATA_DIR, 'noisy_data.dat')
    if not os.path.exists(data_file):
        print(f"Warning: {data_file} not found. Run test_funcUtils first.")
        return
    
    data = np.loadtxt(data_file)
    x = data[:, 0]
    noisy_a_01 = data[:, 1]
    noisy_a_05 = data[:, 2]
    noisy_b_01 = data[:, 3]
    noisy_b_05 = data[:, 4]
    
    # Calculate true function values for comparison
    true_a = 2 * x
    true_b = 2*x - 10*x**5 + 15*x**10
    
    plt.figure(figsize=(15, 10))
    
    # Function A plots
    plt.subplot(2, 2, 1)
    plt.plot(x, true_a, 'b-', linewidth=2, label='True Function A', alpha=0.8)
    plt.scatter(x, noisy_a_01, c='lightblue', s=20, alpha=0.7, label='Noisy (σ=0.1)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Function A with Low Noise (σ=0.1)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(x, true_a, 'b-', linewidth=2, label='True Function A', alpha=0.8)
    plt.scatter(x, noisy_a_05, c='blue', s=20, alpha=0.7, label='Noisy (σ=0.5)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Function A with High Noise (σ=0.5)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Function B plots
    plt.subplot(2, 2, 3)
    plt.plot(x, true_b, 'r-', linewidth=2, label='True Function B', alpha=0.8)
    plt.scatter(x, noisy_b_01, c='lightcoral', s=20, alpha=0.7, label='Noisy (σ=0.1)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Function B with Low Noise (σ=0.1)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot(x, true_b, 'r-', linewidth=2, label='True Function B', alpha=0.8)
    plt.scatter(x, noisy_b_05, c='darkred', s=20, alpha=0.7, label='Noisy (σ=0.5)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Function B with High Noise (σ=0.5)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    output_file = os.path.join(DATA_DIR, 'noisy_data.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ Noisy data plotted and saved as {output_file}")

def plot_polynomial_data():
    """Plot various polynomial functions"""
    data_file = os.path.join(DATA_DIR, 'polynomial_data.dat')
    if not os.path.exists(data_file):
        print(f"Warning: {data_file} not found. Run test_funcUtils first.")
        return
    
    data = np.loadtxt(data_file)
    x = data[:, 0]
    linear = data[:, 1]
    quadratic = data[:, 2]
    cubic = data[:, 3]
    quintic = data[:, 4]
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(x, linear, 'g-', linewidth=2, label='Linear: 1 + 2x')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Linear Polynomial')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(x, quadratic, 'b-', linewidth=2, label='Quadratic: 1 + x²')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Quadratic Polynomial')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(x, cubic, 'r-', linewidth=2, label='Cubic: x + x³')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Cubic Polynomial')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot(x, quintic, 'm-', linewidth=2, label='Quintic: x - 0.5x³ + 0.1x⁵')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Quintic Polynomial')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    output_file = os.path.join(DATA_DIR, 'polynomial_data.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ Polynomial data plotted and saved as {output_file}")

def plot_noise_statistics():
    """Plot noise generation statistics"""
    data_file = os.path.join(DATA_DIR, 'noise_stats.dat')
    if not os.path.exists(data_file):
        print(f"Warning: {data_file} not found. Run test_funcUtils first.")
        return
    
    data = np.loadtxt(data_file)
    expected_stddev = data[:, 0]
    measured_mean = data[:, 1]
    measured_stddev = data[:, 2]
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(expected_stddev, measured_mean, 'bo-', label='Measured Mean')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.7, label='Expected Mean (0)')
    plt.xlabel('Expected Standard Deviation')
    plt.ylabel('Measured Mean')
    plt.title('Noise Generation: Mean Values')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(expected_stddev, measured_stddev, 'ro-', label='Measured Std Dev')
    plt.plot(expected_stddev, expected_stddev, 'k--', alpha=0.7, label='Expected Std Dev')
    plt.xlabel('Expected Standard Deviation')
    plt.ylabel('Measured Standard Deviation')
    plt.title('Noise Generation: Standard Deviation')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    output_file = os.path.join(DATA_DIR, 'noise_statistics.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ Noise statistics plotted and saved as {output_file}")

def plot_comparison():
    """Create a comparison plot showing hidden functions vs polynomials"""
    hidden_file = os.path.join(DATA_DIR, 'hidden_functions.dat')
    poly_file = os.path.join(DATA_DIR, 'polynomial_data.dat')
    
    if not os.path.exists(hidden_file) or not os.path.exists(poly_file):
        print(f"Warning: Required data files not found in {DATA_DIR}. Run test_funcUtils first.")
        return
    
    hidden_data = np.loadtxt(hidden_file)
    poly_data = np.loadtxt(poly_file)
    
    x_hidden = hidden_data[:, 0]
    hidden_a = hidden_data[:, 1]
    hidden_b = hidden_data[:, 2]
    
    x_poly = poly_data[:, 0]
    linear = poly_data[:, 1]
    cubic = poly_data[:, 3]
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(x_hidden, hidden_a, 'b-', linewidth=2, label='hiddenFunctionA(x)')
    plt.plot(x_poly, linear, 'g--', linewidth=2, label='Linear poly (1+2x)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Function A vs Linear Polynomial')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(x_hidden, hidden_b, 'r-', linewidth=2, label='hiddenFunctionB(x)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Complex Hidden Function B')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(x_poly, cubic, 'm-', linewidth=2, label='Cubic poly (x+x³)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Cubic Polynomial')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    output_file = os.path.join(DATA_DIR, 'function_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ Function comparison plotted and saved as {output_file}")

def main():
    """Main plotting function"""
    print("=== funcUtils Test Data Plotter ===\n")
    
    # Check if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as e:
        print(f"Error: Required packages not found: {e}")
        print("Please install: pip install matplotlib numpy")
        sys.exit(1)
    
    # Ensure output directory exists
    ensure_output_dir()
    
    # Set matplotlib style
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 10
    
    # Generate all plots
    plot_hidden_functions()
    plot_noisy_data()
    plot_polynomial_data()
    plot_noise_statistics()
    plot_comparison()
    
    print("\n=== All Plots Generated ===")
    print(f"Generated files in {DATA_DIR}/ directory:")
    print("  - hidden_functions.dat & hidden_functions.png")
    print("  - noisy_data.dat & noisy_data.png") 
    print("  - polynomial_data.dat & polynomial_data.png")
    print("  - noise_stats.dat & noise_statistics.png")
    print("  - function_comparison.png")

if __name__ == "__main__":
    main()