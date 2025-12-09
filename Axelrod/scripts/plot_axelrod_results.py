#!/usr/bin/env python3
"""
Create 2x2 subplots for Axelrod model parameter sweeps.
Visualizes cultural dynamics across different network and feature configurations.

Usage:
  python scripts/plot_axelrod_results.py
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_sweep_data(filepath):
    """Load sweep results from CSV."""
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Warning: {filepath} not found")
        return None


def create_figures(data_dir='data', out_dir='figures'):
    """Create comprehensive 2x2 subplot figures."""
    os.makedirs(out_dir, exist_ok=True)
    
    # Load all sweep data
    neighbors_df = load_sweep_data(os.path.join(data_dir, 'sweep_neighbors_per_node.csv'))
    rewiring_df = load_sweep_data(os.path.join(data_dir, 'sweep_rewiring_prob.csv'))
    features_df = load_sweep_data(os.path.join(data_dir, 'sweep_num_features.csv'))
    feature_dim_df = load_sweep_data(os.path.join(data_dir, 'sweep_feature_dim.csv'))
    
    if neighbors_df is None or rewiring_df is None or features_df is None or feature_dim_df is None:
        print("Error: Could not load all required CSV files")
        return
    
    # Sort by parameter value for consistent plotting
    neighbors_df = neighbors_df.sort_values('param_value').reset_index(drop=True)
    rewiring_df = rewiring_df.sort_values('param_value').reset_index(drop=True)
    features_df = features_df.sort_values('param_value').reset_index(drop=True)
    feature_dim_df = feature_dim_df.sort_values('param_value').reset_index(drop=True)
    
    # ==================== FIGURE 1: Network Structure Effects ====================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Network Structure Effects on Cultural Dynamics', fontsize=16, fontweight='bold')
    
    # Plot 1: Neighbors per Node - Cultural Fragmentation
    ax = axes[0, 0]
    ax.plot(neighbors_df['param_value'], neighbors_df['num_cultures'], 
            marker='o', linewidth=2.5, markersize=8, color='#2E86AB', label='Num Cultures')
    ax.set_xlabel('Neighbors per Node', fontweight='bold')
    ax.set_ylabel('Number of Distinct Cultures', fontweight='bold', color='#2E86AB')
    ax.set_title('Cultural Fragmentation vs Network Connectivity', fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(axis='y', labelcolor='#2E86AB')
    ax.set_facecolor('#f8f9fa')
    
    # Plot 2: Neighbors per Node - Convergence
    ax = axes[0, 1]
    ax.plot(neighbors_df['param_value'], neighbors_df['largest_culture_fraction'], 
            marker='s', linewidth=2.5, markersize=8, color='#A23B72', label='Largest Culture')
    ax.set_xlabel('Neighbors per Node', fontweight='bold')
    ax.set_ylabel('Largest Culture Fraction', fontweight='bold', color='#A23B72')
    ax.set_title('Convergence to Dominant Culture', fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(axis='y', labelcolor='#A23B72')
    ax.set_facecolor('#f8f9fa')
    
    # Plot 3: Neighbors per Node - Entropy
    ax = axes[1, 0]
    ax.plot(neighbors_df['param_value'], neighbors_df['entropy'], 
            marker='^', linewidth=2.5, markersize=8, color='#F18F01', label='Entropy')
    ax.set_xlabel('Neighbors per Node', fontweight='bold')
    ax.set_ylabel('Cultural Diversity (Entropy)', fontweight='bold', color='#F18F01')
    ax.set_title('Entropy vs Network Connectivity', fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(axis='y', labelcolor='#F18F01')
    ax.set_facecolor('#f8f9fa')
    
    # Plot 4: Neighbors per Node - Edge Homophily
    ax = axes[1, 1]
    ax.plot(neighbors_df['param_value'], neighbors_df['edge_homophily'], 
            marker='D', linewidth=2.5, markersize=8, color='#C73E1D', label='Edge Homophily')
    ax.set_xlabel('Neighbors per Node', fontweight='bold')
    ax.set_ylabel('Edge Homophily', fontweight='bold', color='#C73E1D')
    ax.set_title('Network Assortivity by Culture', fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(axis='y', labelcolor='#C73E1D')
    ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'figure1_network_effects.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved {os.path.join(out_dir, 'figure1_network_effects.png')}")
    plt.close()
    
    # ==================== FIGURE 2: Small-World Effects ====================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Small-World Network Properties (Rewiring Probability)', fontsize=16, fontweight='bold')
    
    # Plot 1: Rewiring - Num Cultures
    ax = axes[0, 0]
    ax.plot(rewiring_df['param_value'], rewiring_df['num_cultures'], 
            marker='o', linewidth=2.5, markersize=8, color='#2E86AB')
    ax.set_xlabel('Rewiring Probability', fontweight='bold')
    ax.set_ylabel('Number of Distinct Cultures', fontweight='bold', color='#2E86AB')
    ax.set_title('Cultural Fragmentation vs Disorder', fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(axis='y', labelcolor='#2E86AB')
    ax.set_facecolor('#f8f9fa')
    
    # Plot 2: Rewiring - Largest Culture
    ax = axes[0, 1]
    ax.plot(rewiring_df['param_value'], rewiring_df['largest_culture_fraction'], 
            marker='s', linewidth=2.5, markersize=8, color='#A23B72')
    ax.set_xlabel('Rewiring Probability', fontweight='bold')
    ax.set_ylabel('Largest Culture Fraction', fontweight='bold', color='#A23B72')
    ax.set_title('Convergence Suppression by Rewiring', fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(axis='y', labelcolor='#A23B72')
    ax.set_facecolor('#f8f9fa')
    
    # Plot 3: Rewiring - Entropy
    ax = axes[1, 0]
    ax.plot(rewiring_df['param_value'], rewiring_df['entropy'], 
            marker='^', linewidth=2.5, markersize=8, color='#F18F01')
    ax.set_xlabel('Rewiring Probability', fontweight='bold')
    ax.set_ylabel('Entropy', fontweight='bold', color='#F18F01')
    ax.set_title('Diversity Increase with Disorder', fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(axis='y', labelcolor='#F18F01')
    ax.set_facecolor('#f8f9fa')
    
    # Plot 4: Rewiring - Edge Homophily
    ax = axes[1, 1]
    ax.plot(rewiring_df['param_value'], rewiring_df['edge_homophily'], 
            marker='D', linewidth=2.5, markersize=8, color='#C73E1D')
    ax.set_xlabel('Rewiring Probability', fontweight='bold')
    ax.set_ylabel('Edge Homophily', fontweight='bold', color='#C73E1D')
    ax.set_title('Homophily Breakdown with Random Edges', fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(axis='y', labelcolor='#C73E1D')
    ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'figure2_rewiring_effects.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved {os.path.join(out_dir, 'figure2_rewiring_effects.png')}")
    plt.close()
    
    # ==================== FIGURE 3: Feature Space Dimensionality ====================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Feature Space Effects on Cultural Diversity', fontsize=16, fontweight='bold')
    
    # Plot 1: Num Features - Num Cultures
    ax = axes[0, 0]
    ax.plot(features_df['param_value'], features_df['num_cultures'], 
            marker='o', linewidth=2.5, markersize=8, color='#2E86AB')
    ax.set_xlabel('Number of Features', fontweight='bold')
    ax.set_ylabel('Number of Distinct Cultures', fontweight='bold', color='#2E86AB')
    ax.set_title('Explosive Fragmentation with Feature Dimensionality', fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(axis='y', labelcolor='#2E86AB')
    ax.set_facecolor('#f8f9fa')
    ax.set_yscale('log')
    
    # Plot 2: Num Features - Largest Culture
    ax = axes[0, 1]
    ax.plot(features_df['param_value'], features_df['largest_culture_fraction'], 
            marker='s', linewidth=2.5, markersize=8, color='#A23B72')
    ax.set_xlabel('Number of Features', fontweight='bold')
    ax.set_ylabel('Largest Culture Fraction', fontweight='bold', color='#A23B72')
    ax.set_title('Collapse of Dominant Culture', fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(axis='y', labelcolor='#A23B72')
    ax.set_facecolor('#f8f9fa')
    ax.set_yscale('log')
    
    # Plot 3: Num Features - Entropy
    ax = axes[1, 0]
    ax.plot(features_df['param_value'], features_df['entropy'], 
            marker='^', linewidth=2.5, markersize=8, color='#F18F01')
    ax.set_xlabel('Number of Features', fontweight='bold')
    ax.set_ylabel('Entropy', fontweight='bold', color='#F18F01')
    ax.set_title('Maximum Diversity at High Feature Count', fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(axis='y', labelcolor='#F18F01')
    ax.set_facecolor('#f8f9fa')
    
    # Plot 4: Num Features - Fragmentation Index
    ax = axes[1, 1]
    ax.plot(features_df['param_value'], features_df['fragmentation'], 
            marker='D', linewidth=2.5, markersize=8, color='#C73E1D')
    ax.set_xlabel('Number of Features', fontweight='bold')
    ax.set_ylabel('Fragmentation Index', fontweight='bold', color='#C73E1D')
    ax.set_title('Complete Fragmentation with Multiple Features', fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(axis='y', labelcolor='#C73E1D')
    ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'figure3_feature_count_effects.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved {os.path.join(out_dir, 'figure3_feature_count_effects.png')}")
    plt.close()
    
    # ==================== FIGURE 4: Feature Value Space ====================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Feature Dimension Effects on Cultural Outcomes', fontsize=16, fontweight='bold')
    
    # Plot 1: Feature Dim - Num Cultures
    ax = axes[0, 0]
    ax.plot(feature_dim_df['param_value'], feature_dim_df['num_cultures'], 
            marker='o', linewidth=2.5, markersize=8, color='#2E86AB')
    ax.set_xlabel('Feature Dimension (# Values per Feature)', fontweight='bold')
    ax.set_ylabel('Number of Distinct Cultures', fontweight='bold', color='#2E86AB')
    ax.set_title('Fragmentation with Feature Value Space', fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(axis='y', labelcolor='#2E86AB')
    ax.set_facecolor('#f8f9fa')
    
    # Plot 2: Feature Dim - Largest Culture
    ax = axes[0, 1]
    ax.plot(feature_dim_df['param_value'], feature_dim_df['largest_culture_fraction'], 
            marker='s', linewidth=2.5, markersize=8, color='#A23B72')
    ax.set_xlabel('Feature Dimension (# Values per Feature)', fontweight='bold')
    ax.set_ylabel('Largest Culture Fraction', fontweight='bold', color='#A23B72')
    ax.set_title('Convergence Suppression with Value Space', fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(axis='y', labelcolor='#A23B72')
    ax.set_facecolor('#f8f9fa')
    
    # Plot 3: Feature Dim - Entropy
    ax = axes[1, 0]
    ax.plot(feature_dim_df['param_value'], feature_dim_df['entropy'], 
            marker='^', linewidth=2.5, markersize=8, color='#F18F01')
    ax.set_xlabel('Feature Dimension (# Values per Feature)', fontweight='bold')
    ax.set_ylabel('Entropy', fontweight='bold', color='#F18F01')
    ax.set_title('Rapid Saturation of Diversity', fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(axis='y', labelcolor='#F18F01')
    ax.set_facecolor('#f8f9fa')
    
    # Plot 4: Feature Dim - Edge Homophily
    ax = axes[1, 1]
    ax.plot(feature_dim_df['param_value'], feature_dim_df['edge_homophily'], 
            marker='D', linewidth=2.5, markersize=8, color='#C73E1D')
    ax.set_xlabel('Feature Dimension (# Values per Feature)', fontweight='bold')
    ax.set_ylabel('Edge Homophily', fontweight='bold', color='#C73E1D')
    ax.set_title('Homophily vs Feature Space Complexity', fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(axis='y', labelcolor='#C73E1D')
    ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'figure4_feature_dimension_effects.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved {os.path.join(out_dir, 'figure4_feature_dimension_effects.png')}")
    plt.close()
    
    # ==================== FIGURE 5: Comparative Overview ====================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Comparative Analysis: Key Metrics Across All Sweeps', fontsize=16, fontweight='bold')
    
    # Normalize param values for comparison
    n_neighbors_norm = neighbors_df['param_value'] / neighbors_df['param_value'].max()
    n_rewiring_norm = rewiring_df['param_value'] / rewiring_df['param_value'].max()
    n_features_norm = features_df['param_value'] / features_df['param_value'].max()
    n_feature_dim_norm = feature_dim_df['param_value'] / feature_dim_df['param_value'].max()
    
    # Plot 1: Number of Cultures (all normalized)
    ax = axes[0, 0]
    ax.plot(n_neighbors_norm, neighbors_df['num_cultures'], marker='o', label='Neighbors/Node', linewidth=2.5)
    ax.plot(n_rewiring_norm, rewiring_df['num_cultures'], marker='s', label='Rewiring Prob', linewidth=2.5)
    ax.plot(n_features_norm, features_df['num_cultures'], marker='^', label='Num Features', linewidth=2.5)
    ax.plot(n_feature_dim_norm, feature_dim_df['num_cultures'], marker='D', label='Feature Dim', linewidth=2.5)
    ax.set_xlabel('Normalized Parameter Value', fontweight='bold')
    ax.set_ylabel('Number of Cultures', fontweight='bold')
    ax.set_title('Cultural Fragmentation Comparison', fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_facecolor('#f8f9fa')
    
    # Plot 2: Largest Culture Fraction
    ax = axes[0, 1]
    ax.plot(n_neighbors_norm, neighbors_df['largest_culture_fraction'], marker='o', label='Neighbors/Node', linewidth=2.5)
    ax.plot(n_rewiring_norm, rewiring_df['largest_culture_fraction'], marker='s', label='Rewiring Prob', linewidth=2.5)
    ax.plot(n_features_norm, features_df['largest_culture_fraction'], marker='^', label='Num Features', linewidth=2.5)
    ax.plot(n_feature_dim_norm, feature_dim_df['largest_culture_fraction'], marker='D', label='Feature Dim', linewidth=2.5)
    ax.set_xlabel('Normalized Parameter Value', fontweight='bold')
    ax.set_ylabel('Largest Culture Fraction', fontweight='bold')
    ax.set_title('Convergence to Dominant Culture', fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_facecolor('#f8f9fa')
    ax.set_yscale('log')
    
    # Plot 3: Entropy
    ax = axes[1, 0]
    ax.plot(n_neighbors_norm, neighbors_df['entropy'], marker='o', label='Neighbors/Node', linewidth=2.5)
    ax.plot(n_rewiring_norm, rewiring_df['entropy'], marker='s', label='Rewiring Prob', linewidth=2.5)
    ax.plot(n_features_norm, features_df['entropy'], marker='^', label='Num Features', linewidth=2.5)
    ax.plot(n_feature_dim_norm, feature_dim_df['entropy'], marker='D', label='Feature Dim', linewidth=2.5)
    ax.set_xlabel('Normalized Parameter Value', fontweight='bold')
    ax.set_ylabel('Entropy', fontweight='bold')
    ax.set_title('Cultural Diversity Across Parameters', fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_facecolor('#f8f9fa')
    
    # Plot 4: Edge Homophily
    ax = axes[1, 1]
    ax.plot(n_neighbors_norm, neighbors_df['edge_homophily'], marker='o', label='Neighbors/Node', linewidth=2.5)
    ax.plot(n_rewiring_norm, rewiring_df['edge_homophily'], marker='s', label='Rewiring Prob', linewidth=2.5)
    ax.plot(n_features_norm, features_df['edge_homophily'], marker='^', label='Num Features', linewidth=2.5)
    ax.plot(n_feature_dim_norm, feature_dim_df['edge_homophily'], marker='D', label='Feature Dim', linewidth=2.5)
    ax.set_xlabel('Normalized Parameter Value', fontweight='bold')
    ax.set_ylabel('Edge Homophily', fontweight='bold')
    ax.set_title('Network Assortivity Comparison', fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'figure5_comparative_overview.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved {os.path.join(out_dir, 'figure5_comparative_overview.png')}")
    plt.close()
    
    print("\n" + "="*60)
    print("All figures generated successfully!")
    print("="*60)
    print("\nKey Observations:")
    print("• Figure 1: Network connectivity weakly affects cultural dynamics")
    print("• Figure 2: Disorder (rewiring) increases fragmentation & diversity")
    print("• Figure 3: CRITICAL EFFECT - More features → exponential fragmentation")
    print("• Figure 4: Feature value space causes rapid diversity saturation")
    print("• Figure 5: Feature dimensionality dominates all other parameters")


if __name__ == '__main__':
    create_figures()
