import matplotlib.pyplot as plt
import numpy as np

def fileReader(filepath):
    data = np.loadtxt(filepath)
    
    uniform_data = data[:, 0]
    datapoints_a = data[:, 1] 
    datapoints_b = data[:, 2]
    
    return {
        'uniform_data': uniform_data,
        'datapointsA': datapoints_a,
        'datapointsB': datapoints_b
    }

def fittedDataReader(filepath):
    data = np.loadtxt(filepath, skiprows=1)  # Skip header row
    
    x_pred = data[:, 0]
    fit1_A = data[:, 1]
    fit3_A = data[:, 2]
    fit10_A = data[:, 3]
    fit1_B = data[:, 4]
    fit3_B = data[:, 5]
    fit10_B = data[:, 6]
    
    return {
        'x_pred': x_pred,
        'fit1_A': fit1_A,
        'fit3_A': fit3_A,
        'fit10_A': fit10_A,
        'fit1_B': fit1_B,
        'fit3_B': fit3_B,
        'fit10_B': fit10_B
    }

def dataPlotter(data):
    plt.figure(figsize=(10, 6))
    plt.plot(data['uniform_data'], data['datapointsA'], "o", label='Datapoints A', color='orange')
    plt.plot(data['uniform_data'], data['datapointsB'], "o", label='Datapoints B', color='green')
    plt.xlabel('Index')
    plt.ylabel('Datapoints')
    plt.title('Datapoints A and B from Uniform Data')
    plt.legend()
    plt.grid(True)
    plt.show()

def plotWithFits():
    # Read original data
    original_data = fileReader('output.txt')
    
    # Read fitted data
    fitted_data = fittedDataReader('fitted_output.txt')
    
    # Calculate boundaries based on original data with some padding
    x_min = min(original_data['uniform_data'])
    x_max = max(original_data['uniform_data'])
    x_padding = (x_max - x_min) * 0.1  # 10% padding
    
    y_min_A = min(original_data['datapointsA'])
    y_max_A = max(original_data['datapointsA'])
    y_padding_A = (y_max_A - y_min_A) * 0.15  # 15% padding
    
    y_min_B = min(original_data['datapointsB'])
    y_max_B = max(original_data['datapointsB'])
    y_padding_B = (y_max_B - y_min_B) * 0.15  # 15% padding
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot for Dataset A
    ax1.plot(original_data['uniform_data'], original_data['datapointsA'], 'o', 
             label='Original Data A', color='orange', markersize=8, alpha=0.7)
    ax1.plot(fitted_data['x_pred'], fitted_data['fit1_A'], '-', 
             label='Polynomial Order 1', color='blue', linewidth=2)
    ax1.plot(fitted_data['x_pred'], fitted_data['fit3_A'], '-', 
             label='Polynomial Order 3', color='red', linewidth=2)
    ax1.plot(fitted_data['x_pred'], fitted_data['fit10_A'], '-', 
             label='Polynomial Order 10', color='purple', linewidth=2)
    
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Dataset A: Original Data and Polynomial Fits')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Set boundaries for Dataset A
    ax1.set_xlim(x_min - x_padding, x_max + x_padding)
    ax1.set_ylim(y_min_A - y_padding_A, y_max_A + y_padding_A)
    
    # Plot for Dataset B
    ax2.plot(original_data['uniform_data'], original_data['datapointsB'], 'o', 
             label='Original Data B', color='green', markersize=8, alpha=0.7)
    ax2.plot(fitted_data['x_pred'], fitted_data['fit1_B'], '-', 
             label='Polynomial Order 1', color='blue', linewidth=2)
    ax2.plot(fitted_data['x_pred'], fitted_data['fit3_B'], '-', 
             label='Polynomial Order 3', color='red', linewidth=2)
    ax2.plot(fitted_data['x_pred'], fitted_data['fit10_B'], '-', 
             label='Polynomial Order 10', color='purple', linewidth=2)
    
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Dataset B: Original Data and Polynomial Fits')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Set boundaries for Dataset B
    ax2.set_xlim(x_min - x_padding, x_max + x_padding)
    ax2.set_ylim(y_min_B - y_padding_B, y_max_B + y_padding_B)
    
    plt.tight_layout()
    plt.show()

def exOnePlotter():
    return

def exTwoPlotter():
    data = fileReader('output.txt')
    dataPlotter(data)

def exThreePlotter():
    plotWithFits()

if __name__ == "__main__":
    # Run exercise 2 plotter (original data)
    print("Plotting original data...")
    exTwoPlotter()
    
    # Run exercise 3 plotter (data with polynomial fits)
    print("Plotting data with polynomial fits...")
    exThreePlotter()