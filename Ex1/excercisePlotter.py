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

def exOnePlotter():
    return

def exTwoPlotter():
    data = fileReader('output.txt')
    plt.figure(figsize=(10, 6))
    plt.plot(data['uniform_data'], label='Uniform Data', color='blue')
    plt.plot(data['datapointsA'], label='Datapoints A', color='orange')
    plt.plot(data['datapointsB'], label='Datapoints B', color='green')
    plt.xlabel('Index')
    plt.ylabel('Datapoints')
    plt.title('Datapoints A and B from Uniform Data')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    exTwoPlotter()