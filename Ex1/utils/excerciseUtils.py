import numpy as np
import matplotlib.pyplot as plt

ORDER_COLORS = {1: 'blue', 3: 'red', 10: 'purple'}
DATASET_SCATTER_COLORS = {'A': 'orange', 'B': 'green'}

def _analyzeFileStructure(filepath):
    """
    Analyze the structure of the file to detect:
    1. Presence of test data section
    2. Location of fitted data header
    3. Structure of different sections
    """
    structure = {
        'has_test_data': False,
        'test_data_start': None,
        'test_data_end': None,
        'fitted_data_header_line': None,
        'fitted_data_start': None,
        'parameter_lines': []
    }
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Look for the fitted data header
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Check for parameter lines (comments starting with #)
        if line.startswith('#') and ('Order' in line or 'Fitted parameters' in line):
            structure['parameter_lines'].append(i)
        
        # Check for test data section header
        if line.startswith('# Test points'):
            structure['has_test_data'] = True
            structure['test_data_start'] = i + 1  # Data starts after header
        
        # Check for fitted data header
        if 'x_pred' in line and 'fit1_A' in line and 'fit3_A' in line:
            structure['fitted_data_header_line'] = i
            structure['fitted_data_start'] = i + 1
            # If we found test data before, mark where it ends
            if structure['has_test_data'] and structure['test_data_end'] is None:
                structure['test_data_end'] = i - 1  # Test data ends before this line
            break
    
    return structure

def fileReader(filepath):
    """
    Read data from files that may contain:
    1. Only original data (like output.txt)
    2. Test data section + fitted data (like fitted_output_with_test.txt)
    
    Returns the original/test data in the same format regardless of file structure
    """
    structure = _analyzeFileStructure(filepath)
    
    if structure['has_test_data']:
        # File has test data section - read it
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Extract test data lines (skip the header)
        test_data_lines = []
        start_idx = structure['test_data_start']
        end_idx = structure['test_data_end']
        
        for i in range(start_idx, end_idx + 1):
            line = lines[i].strip()
            if line and not line.startswith('#'):  # Skip empty lines and comments
                test_data_lines.append(line)
        
        # Parse the test data
        data_array = []
        for line in test_data_lines:
            values = [float(x) for x in line.split()]
            if len(values) >= 3:  # Ensure we have at least 3 columns
                data_array.append(values)
        
        data = np.array(data_array)
        
        uniform_data = data[:, 0]
        datapoints_a = data[:, 1] 
        datapoints_b = data[:, 2]
        
    else:
        # File has only original data - use original method
        data = np.loadtxt(filepath, skiprows=1)  # Skip header row
        
        uniform_data = data[:, 0]
        datapoints_a = data[:, 1] 
        datapoints_b = data[:, 2]
    
    return {
        'uniform_data': uniform_data,
        'datapointsA': datapoints_a,
        'datapointsB': datapoints_b
    }

def fittedDataReader(filepath, skiprow_index=None):
    """
    Read fitted data from files with automatic header detection.
    Works with both fitted_output.txt and fitted_output_with_test.txt formats.
    
    Args:
        filepath: Path to the file
        skiprow_index: Manual override for number of rows to skip (optional)
    """
    if skiprow_index is None:
        # Automatically detect the structure
        structure = _analyzeFileStructure(filepath)
        if structure['fitted_data_start'] is not None:
            skiprow_index = structure['fitted_data_start']
        else:
            skiprow_index = 8
    
    data = np.loadtxt(filepath, skiprows=skiprow_index)
    
    # Ensure we have the expected number of columns
    if data.shape[1] < 7:
        raise ValueError(f"Expected at least 7 columns in fitted data, but found {data.shape[1]}")
    
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

def parameterReader(filepath):
    """Read fitted parameters from comment lines in the fitted output file"""
    params = {
        'params1_A': [],
        'params3_A': [],
        'params10_A': [],
        'params1_B': [],
        'params3_B': [],
        'params10_B': []
    }
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line.startswith('#'):
                break  
            
            if 'Order 1 A:' in line:
                # Extract numbers after "Order 1 A:"
                param_str = line.split('Order 1 A:')[1].strip()
                params['params1_A'] = [float(x) for x in param_str.split()]
            elif 'Order 3 A:' in line:
                param_str = line.split('Order 3 A:')[1].strip()
                params['params3_A'] = [float(x) for x in param_str.split()]
            elif 'Order 10 A:' in line:
                param_str = line.split('Order 10 A:')[1].strip()
                params['params10_A'] = [float(x) for x in param_str.split()]
            elif 'Order 1 B:' in line:
                param_str = line.split('Order 1 B:')[1].strip()
                params['params1_B'] = [float(x) for x in param_str.split()]
            elif 'Order 3 B:' in line:
                param_str = line.split('Order 3 B:')[1].strip()
                params['params3_B'] = [float(x) for x in param_str.split()]
            elif 'Order 10 B:' in line:
                param_str = line.split('Order 10 B:')[1].strip()
                params['params10_B'] = [float(x) for x in param_str.split()]
    
    return params

def plotParameters(params, ax, dataset_label):
    """Plot fitted parameters as bar charts"""
    max_order = 10
    
    orders = ['Order 1', 'Order 3', 'Order 10']
    colors = ['blue', 'red', 'purple']
    
    if dataset_label == 'A':
        param_sets = [params['params1_A'], params['params3_A'], params['params10_A']]
    else:  # dataset_label == 'B'
        param_sets = [params['params1_B'], params['params3_B'], params['params10_B']]
    
    x_positions = []
    param_values = []
    order_labels = []
    bar_colors = []
    
    current_x = 0
    for i, (order_name, param_set, color) in enumerate(zip(orders, param_sets, colors)):
        for j, param in enumerate(param_set):
            x_positions.append(current_x + j)
            param_values.append(param)
            order_labels.append(f'{order_name}\nParam {j}')
            bar_colors.append(color)
        current_x += len(param_set) + 1  
        
    bars = ax.bar(x_positions, param_values, color=bar_colors, alpha=0.7)
    
    ax.set_xlabel('Parameters')
    ax.set_ylabel('Parameter Values')
    ax.set_title(f'Fitted Parameters - Dataset {dataset_label}')
    ax.grid(True, alpha=0.3, axis='y')
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(order_labels, rotation=45, ha='right', fontsize=8)
    
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.7, label=order) 
                      for order, color in zip(orders, colors)]
    ax.legend(handles=legend_elements, loc='upper right')
    
    return ax

def dataPlotter(data, save_plot=False, filename='exercise2_plot.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(data['uniform_data'], data['datapointsA'], "o", label='Datapoints A', color='orange')
    plt.plot(data['uniform_data'], data['datapointsB'], "o", label='Datapoints B', color='green')
    plt.xlabel('Index')
    plt.ylabel('Datapoints')
    plt.title('Datapoints A and B from Uniform Data')
    plt.legend()
    plt.grid(True)
    
    if save_plot:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {filename}")
    else:
        plt.show()

def plotWithFits(save_plot=False, filename='exercise3_plot.png'):
    original_data = fileReader('output.txt')
    
    fitted_data = fittedDataReader('fitted_output.txt')
    
    params = parameterReader('fitted_output.txt')
    
    x_min = min(original_data['uniform_data'])
    x_max = max(original_data['uniform_data'])
    x_padding = (x_max - x_min) * 0.1  # 10% padding
    
    y_min_A = min(original_data['datapointsA'])
    y_max_A = max(original_data['datapointsA'])
    y_padding_A = (y_max_A - y_min_A) * 0.15  # 15% padding
    
    y_min_B = min(original_data['datapointsB'])
    y_max_B = max(original_data['datapointsB'])
    y_padding_B = (y_max_B - y_min_B) * 0.15  # 15% padding
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Top row: Plot fitted curves for Dataset A and B
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
    
    ax1.set_xlim(x_min - x_padding, x_max + x_padding)
    ax1.set_ylim(y_min_A - y_padding_A, y_max_A + y_padding_A)
    
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
    
    # Bottom row: Plot fitted parameters for Dataset A and B
    plotParameters(params, ax3, 'A')
    plotParameters(params, ax4, 'B')
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {filename}")
    else:
        plt.show()