import os
from utils.comparisonUtils import plotComparisonFigure

def comparisonPlotter():
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
    comparisonPlotter()