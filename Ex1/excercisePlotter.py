import os
import matplotlib.pyplot as plt
import numpy as np

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


def comparisonFileReader(filepath):
    """Read comparison output generated by exPointFive."""
    data_x, data_a, data_b = [], [], []
    methods = {}
    orders = []
    current_method = None
    prediction_mapping = []

    with open(filepath, 'r') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith('#'):
                if line.startswith('# Method:'):
                    current_method = line.split(':', 1)[1].strip()
                    methods[current_method] = {
                        'params': {'A': {}, 'B': {}},
                        'predictions': {'A': {}, 'B': {}},
                        'x_pred': []
                    }
                    prediction_mapping = []
                elif line.startswith('# Order') and current_method is not None:
                    header, values_part = line.split(':', 1)
                    header_tokens = header.split()
                    if len(header_tokens) < 4:
                        continue
                    order = int(header_tokens[2])
                    dataset_label = header_tokens[3]
                    values_str = values_part.strip()
                    values = [float(x) for x in values_str.split()] if values_str else []
                    methods[current_method]['params'][dataset_label][order] = values
                    if order not in orders:
                        orders.append(order)
                else:
                    continue
            else:
                tokens = line.split()
                if not tokens:
                    continue
                if tokens[0] == 'x':
                    continue
                if tokens[0] == 'x_pred':
                    if current_method is None:
                        continue
                    prediction_mapping = []
                    for token in tokens[1:]:
                        if '_' not in token:
                            continue
                        fit_part, dataset_label = token.split('_')
                        order = int(fit_part.replace('fit', ''))
                        prediction_mapping.append((dataset_label, order))
                        if order not in orders:
                            orders.append(order)
                        if order not in methods[current_method]['predictions'][dataset_label]:
                            methods[current_method]['predictions'][dataset_label][order] = []
                    methods[current_method]['x_pred'] = []
                    continue

                values = [float(x) for x in tokens]
                if current_method is None:
                    if len(values) >= 3:
                        data_x.append(values[0])
                        data_a.append(values[1])
                        data_b.append(values[2])
                else:
                    if not prediction_mapping:
                        continue
                    methods[current_method]['x_pred'].append(values[0])
                    for (dataset_label, order), value in zip(prediction_mapping, values[1:]):
                        methods[current_method]['predictions'][dataset_label][order].append(value)

    return {
        'data': {
            'x': np.array(data_x),
            'datapointsA': np.array(data_a),
            'datapointsB': np.array(data_b)
        },
        'methods': methods,
        'orders': sorted(orders)
    }


def _plot_comparison_fit(ax, x_train, y_train, method_data, orders, dataset_label, method_name,
                         x_bounds, y_bounds):
    ax.scatter(x_train, y_train, color=DATASET_SCATTER_COLORS.get(dataset_label, 'gray'),
               alpha=0.7, s=45, label=f'Dataset {dataset_label}')

    x_pred = np.array(method_data['x_pred'])
    for order in orders:
        preds_for_dataset = method_data['predictions'][dataset_label]
        if order not in preds_for_dataset:
            continue
        color = ORDER_COLORS.get(order, plt.cm.tab10(order % 10))
        y_pred = np.array(preds_for_dataset[order])
        ax.plot(x_pred, y_pred, color=color, linewidth=2, label=f'Order {order}')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'{method_name} • Dataset {dataset_label}')
    ax.grid(True, alpha=0.3)
    if x_bounds is not None:
        ax.set_xlim(*x_bounds)
    if y_bounds is not None:
        ax.set_ylim(*y_bounds)
    ax.legend()


def _plot_comparison_parameters(ax, method_data, orders, dataset_label, method_name):
    params_dict = method_data['params'][dataset_label]
    x_positions, heights, colors, labels = [], [], [], []
    current_x = 0

    for order in orders:
        values = params_dict.get(order, [])
        if not values:
            continue
        for idx, val in enumerate(values):
            x_positions.append(current_x)
            heights.append(val)
            colors.append(ORDER_COLORS.get(order, 'gray'))
            labels.append(f'θ{idx}\nord {order}')
            current_x += 1
        current_x += 1  # spacing between orders

    if not x_positions:
        ax.text(0.5, 0.5, 'No parameters', ha='center', va='center')
        ax.axis('off')
        return

    ax.bar(x_positions, heights, color=colors, alpha=0.7)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Parameter values')
    ax.set_title(f'{method_name} • Parameters {dataset_label}')
    ax.grid(True, alpha=0.3, axis='y')

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=ORDER_COLORS.get(order, 'gray'), alpha=0.7, label=f'Order {order}')
        for order in orders if order in params_dict
    ]
    if legend_handles:
        ax.legend(handles=legend_handles, loc='upper right')


def plotComparisonFigure(filepath, title, output_path=None):
    comparison = comparisonFileReader(filepath)
    data = comparison['data']
    methods = list(comparison['methods'].items())
    orders = comparison['orders']

    if len(methods) != 2:
        raise ValueError(f"Expected exactly two methods in comparison file, found {len(methods)}")

    x_train = data['x']
    y_train_A = data['datapointsA']
    y_train_B = data['datapointsB']

    if x_train.size > 0:
        x_min, x_max = x_train.min(), x_train.max()
        x_padding = (x_max - x_min) * 0.1 if x_max != x_min else 0.1
        x_bounds = (x_min - x_padding, x_max + x_padding)
    else:
        x_bounds = None

    def _compute_bounds(values):
        if values.size == 0:
            return None
        y_min, y_max = values.min(), values.max()
        padding = (y_max - y_min) * 0.15 if y_max != y_min else 0.1
        return (y_min - padding, y_max + padding)

    yA_bounds = _compute_bounds(y_train_A)
    yB_bounds = _compute_bounds(y_train_B)

    fig, axes = plt.subplots(4, 2, figsize=(18, 24))

    for idx, (method_name, method_data) in enumerate(methods):
        row_offset = idx * 2
        _plot_comparison_fit(axes[row_offset, 0], x_train, y_train_A, method_data, orders, 'A', method_name, x_bounds, yA_bounds)
        _plot_comparison_fit(axes[row_offset, 1], x_train, y_train_B, method_data, orders, 'B', method_name, x_bounds, yB_bounds)
        _plot_comparison_parameters(axes[row_offset + 1, 0], method_data, orders, 'A', method_name)
        _plot_comparison_parameters(axes[row_offset + 1, 1], method_data, orders, 'B', method_name)

    fig.suptitle(title, fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    plt.show()

def plotParameters(params, ax, dataset_label):
    """Plot fitted parameters as bar charts"""
    max_order = 10
    
    # Prepare data for plotting
    orders = ['Order 1', 'Order 3', 'Order 10']
    colors = ['blue', 'red', 'purple']
    
    if dataset_label == 'A':
        param_sets = [params['params1_A'], params['params3_A'], params['params10_A']]
    else:  # dataset_label == 'B'
        param_sets = [params['params1_B'], params['params3_B'], params['params10_B']]
    
    # Create bar positions
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
        current_x += len(param_set) + 1  # Add space between different orders
    
    # Create the bar plot
    bars = ax.bar(x_positions, param_values, color=bar_colors, alpha=0.7)
    
    # Customize the plot
    ax.set_xlabel('Parameters')
    ax.set_ylabel('Parameter Values')
    ax.set_title(f'Fitted Parameters - Dataset {dataset_label}')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Set x-axis labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(order_labels, rotation=45, ha='right', fontsize=8)
    
    # Add legend
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.7, label=order) 
                      for order, color in zip(orders, colors)]
    ax.legend(handles=legend_elements, loc='upper right')
    
    return ax

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
    
    # Create figure with 2x2 subplots
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
    plt.show()

def exOnePlotter():
    return

def exTwoPlotter():
    data = fileReader('output.txt')
    dataPlotter(data)

def exThreePlotter():
    plotWithFits()

def exFourPlotter():
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
    plt.show()
    
    print(f"Successfully loaded:")
    print(f"- {len(testDataPoints['uniform_data'])} test data points")
    print(f"- {len(fittedData['x_pred'])} fitted data points")

def exFivePlotter():
    """Generate comparison plots for GD vs SGD using exPointFive outputs."""
    comparison_cases = [
        ('gd_vs_sgd_fullbatch.txt', 'GD vs SGD (full batch)', 'gd_vs_sgd_fullbatch.png'),
        ('gd_vs_sgd_minibatch2.txt', 'GD vs SGD (mini-batch = 2)', 'gd_vs_sgd_minibatch2.png')
    ]

    for filepath, title, output_path in comparison_cases:
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found. Run exPointFive() to generate it.")
            continue
        plotComparisonFigure(filepath, title, output_path)

if __name__ == "__main__":
    # Run exercise 2 plotter (original data)
    # print("Plotting original data...")
    # exTwoPlotter()
    
    # Run exercise 3 plotter (data with polynomial fits)
    # print("Plotting data with polynomial fits...")
    # exThreePlotter()

    # Run exercise 4 plotter (data with test points and polynomial fits)
    print("Plotting data with test points and polynomial fits...")
    # exFourPlotter()

    # Run exercise 5 plotter (comparison of GD vs SGD)
    print("Plotting comparison of GD vs SGD...")
    exFivePlotter()