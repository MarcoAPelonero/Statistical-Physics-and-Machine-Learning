import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
from utils.excerciseUtils import ORDER_COLORS, DATASET_SCATTER_COLORS
from utils.excerciseUtils import fileReader, dataPlotter, plotWithFits, fittedDataReader
from utils.comparisonUtils import plotComparisonFigure
from utils.comparisonUtils import plotMethodFigure
from utils.comparisonUtils import plotMethodFitsOnly

def exOnePlotter():
    return

def exTwoPlotter(save_plot=False):
    data = fileReader('output.txt')
    dataPlotter(data, save_plot=save_plot, filename='exercise2_plot.png')

def exThreePlotter(save_plot=False):
    plotWithFits(save_plot=save_plot, filename='exercise3_plot.png')

def exFourPlotter(save_plot=False):
    """
    Example function demonstrating compatibility with fitted_output_with_test.txt
    """
    testDataPoints = fileReader('fitted_output_with_test.txt')
    
    fittedData = fittedDataReader('fitted_output_with_test.txt')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.scatter(testDataPoints['uniform_data'], testDataPoints['datapointsA'], 
                color='orange', alpha=0.8, s=50, label='Test Data A')
    ax1.plot(fittedData['x_pred'], fittedData['fit1_A'], '-', 
             label='Polynomial Order 1', color='blue', linewidth=2)
    ax1.plot(fittedData['x_pred'], fittedData['fit3_A'], '-', 
             label='Polynomial Order 3', color='red', linewidth=2)
    ax1.plot(fittedData['x_pred'], fittedData['fit10_A'], '-', 
             label='Polynomial Order 10', color='purple', linewidth=2)
    
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Dataset A: Test Data and Polynomial Fits')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.scatter(testDataPoints['uniform_data'], testDataPoints['datapointsB'], 
                color='green', alpha=0.8, s=50, label='Test Data B')
    ax2.plot(fittedData['x_pred'], fittedData['fit1_B'], '-', 
             label='Polynomial Order 1', color='blue', linewidth=2)
    ax2.plot(fittedData['x_pred'], fittedData['fit3_B'], '-', 
             label='Polynomial Order 3', color='red', linewidth=2)
    ax2.plot(fittedData['x_pred'], fittedData['fit10_B'], '-', 
             label='Polynomial Order 10', color='purple', linewidth=2)
    
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Dataset B: Test Data and Polynomial Fits')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig('exercise4_plot.png', dpi=300, bbox_inches='tight')
        print(f"Plot saved as exercise4_plot.png")
    else:
        plt.show()
    
    print(f"Successfully loaded:")
    print(f"- {len(testDataPoints['uniform_data'])} test data points")
    print(f"- {len(fittedData['x_pred'])} fitted data points")

def exSevenPlotter(save_plot=False):
    filepath = 'exercise7_comparison.txt'
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found. Run exPointSeven() to generate it.")
        return

    output_path = 'exercise7_plot.png' if save_plot else None
    title = 'Exercise 7: GD vs SGD (100 samples)'
    plotComparisonFigure(filepath, title, output_path=output_path)
    if save_plot and output_path:
        print(f"Plot saved as {output_path}")

    # Also produce single-method figures: GD and SGD
    try:
        gd_output = 'exercise7_gd_plot.png' if save_plot else None
        sgd_output = 'exercise7_sgd_plot.png' if save_plot else None
        plotMethodFigure(filepath, 'GD', title='Exercise 7: GD only (100 samples)', output_path=gd_output)
        if save_plot and gd_output:
            print(f"GD-only plot saved as {gd_output}")
        plotMethodFigure(filepath, 'SGD batch=10', title='Exercise 7: SGD only (100 samples)', output_path=sgd_output)
        if save_plot and sgd_output:
            print(f"SGD-only plot saved as {sgd_output}")
    except Exception as e:
        print(f"Could not produce single-method plots for Exercise 7: {e}")

    # Also produce fits-only figures (no weights) with x axis clipped to [0,1]
    try:
        gd_fits_output = 'exercise7_gd_fits_only.png' if save_plot else None
        sgd_fits_output = 'exercise7_sgd_fits_only.png' if save_plot else None
        plotMethodFitsOnly(filepath, 'GD', title='Exercise 7: GD fits only', output_path=gd_fits_output, xlim=(0, 1))
        if save_plot and gd_fits_output:
            print(f"GD fits-only plot saved as {gd_fits_output}")
        plotMethodFitsOnly(filepath, 'SGD batch=10', title='Exercise 7: SGD fits only', output_path=sgd_fits_output, xlim=(0, 1))
        if save_plot and sgd_fits_output:
            print(f"SGD fits-only plot saved as {sgd_fits_output}")
    except Exception as e:
        print(f"Could not produce fits-only plots for Exercise 7: {e}")

def exNinePlotter(save_plot=False):
    filepath = 'exercise9_comparison.txt'
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found. Run exPointNine() to generate it.")
        return

    output_path = 'exercise9_plot.png' if save_plot else None
    title = 'Exercise 9: GD vs SGD (10k samples)'
    plotComparisonFigure(filepath, title, output_path=output_path)
    if save_plot and output_path:
        print(f"Plot saved as {output_path}")

    # Also produce single-method figures: GD and SGD
    try:
        gd_output = 'exercise9_gd_plot.png' if save_plot else None
        sgd_output = 'exercise9_sgd_plot.png' if save_plot else None
        plotMethodFigure(filepath, 'GD', title='Exercise 9: GD only (10k samples)', output_path=gd_output)
        if save_plot and gd_output:
            print(f"GD-only plot saved as {gd_output}")
        plotMethodFigure(filepath, 'SGD batch=100', title='Exercise 9: SGD only (10k samples)', output_path=sgd_output)
        if save_plot and sgd_output:
            print(f"SGD-only plot saved as {sgd_output}")
    except Exception as e:
        print(f"Could not produce single-method plots for Exercise 9: {e}")

    # Also produce fits-only figures (no weights) with x axis clipped to [0,1]
    try:
        gd_fits_output = 'exercise9_gd_fits_only.png' if save_plot else None
        sgd_fits_output = 'exercise9_sgd_fits_only.png' if save_plot else None
        plotMethodFitsOnly(filepath, 'GD', title='Exercise 9: GD fits only', output_path=gd_fits_output, xlim=(0, 1))
        if save_plot and gd_fits_output:
            print(f"GD fits-only plot saved as {gd_fits_output}")
        plotMethodFitsOnly(filepath, 'SGD batch=100', title='Exercise 9: SGD fits only', output_path=sgd_fits_output, xlim=(0, 1))
        if save_plot and sgd_fits_output:
            print(f"SGD fits-only plot saved as {sgd_fits_output}")
    except Exception as e:
        print(f"Could not produce fits-only plots for Exercise 9: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Exercise plotter with configurable options')
    parser.add_argument('-e', '--exercise', type=int, choices=[2, 3, 4, 7, 9], 
                        help='Which exercise to run (2, 3, 4, 7, or 9). Default: run all')
    parser.add_argument('-s', '--save', action='store_true', 
                        help='Save plots as PNG files instead of displaying them')
    
    args = parser.parse_args()
    
    if args.exercise:
        exercises_to_run = [args.exercise]
    else:
        exercises_to_run = [2, 3, 4, 7, 9]  
    for exercise in exercises_to_run:
        if exercise == 2:
            print("Running Exercise 2: Plotting original data...")
            exTwoPlotter(save_plot=args.save)
        elif exercise == 3:
            print("Running Exercise 3: Plotting data with polynomial fits...")
            exThreePlotter(save_plot=args.save)
        elif exercise == 4:
            print("Running Exercise 4: Plotting data with test points and polynomial fits...")
            exFourPlotter(save_plot=args.save)
        elif exercise == 7:
            print("Running Exercise 7: Comparison with test data and true curve...")
            exSevenPlotter(save_plot=args.save)
        elif exercise == 9:
            print("Running Exercise 9: Comparison with test data and true curve...")
            exNinePlotter(save_plot=args.save)
    
    if args.save:
        print(f"\nAll plots saved successfully!")
    else:
        print(f"\nAll plots displayed successfully!")