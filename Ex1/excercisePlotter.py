import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
from utils.excerciseUtils import ORDER_COLORS, DATASET_SCATTER_COLORS
from utils.excerciseUtils import fileReader, dataPlotter, plotWithFits, fittedDataReader

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
    
    # On the 2 axis of the figure plot data point from A and the fitted curves for A, and on the second for B
    
    # Plot A: Test data points and fitted curves for dataset A
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
    
    # Plot B: Test data points and fitted curves for dataset B
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Exercise plotter with configurable options')
    parser.add_argument('-e', '--exercise', type=int, choices=[2, 3, 4], 
                        help='Which exercise to run (2, 3, or 4). Default: run all')
    parser.add_argument('-s', '--save', action='store_true', 
                        help='Save plots as PNG files instead of displaying them')
    
    args = parser.parse_args()
    
    if args.exercise:
        exercises_to_run = [args.exercise]
    else:
        exercises_to_run = [2, 3, 4]  
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
    
    if args.save:
        print(f"\nAll plots saved successfully!")
    else:
        print(f"\nAll plots displayed successfully!")