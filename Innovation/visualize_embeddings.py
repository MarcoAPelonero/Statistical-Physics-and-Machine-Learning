"""
Visualize embeddings over time with dimensionality reduction and clustering.
Creates 2D plots for each year and an animated GIF showing evolution.
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.linalg import orthogonal_procrustes
from pathlib import Path
import imageio
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Parse command line arguments
DARKMODE = "--darkmode" in sys.argv

# Configure styling based on mode
if DARKMODE:
    plt.style.use('dark_background')
    FACECOLOR = 'black'
    TITLE_COLOR = 'white'
    EDGE_COLOR = 'white'
    GRID_COLOR = 'gray'
else:
    plt.style.use('default')
    FACECOLOR = 'white'
    TITLE_COLOR = 'black'
    EDGE_COLOR = 'black'
    GRID_COLOR = 'lightgray'

# Configuration
OUTPUT_DIR = Path("output")
YEARS = list(range(2005, 2021))
N_CLUSTERS = 8  # Number of technological area clusters
RANDOM_STATE = 42
PERPLEXITY = 30
N_ITER = 1000
REFERENCE_YEAR = 2010  # Reference year for Procrustes alignment

# Technology area color palette
COLORS = [
    '#e41a1c',  # Red
    '#377eb8',  # Blue
    '#4daf4a',  # Green
    '#984ea3',  # Purple
    '#ff7f00',  # Orange
    '#ffff33',  # Yellow
    '#a65628',  # Brown
    '#f781bf',  # Pink
]


def load_mesh_names(dataset_path="Dataset.jsonl", max_lines=100000):
    """Load MeSH code to name mapping from Dataset.jsonl"""
    print("Loading MeSH code names from Dataset.jsonl...")
    mesh_names = {}
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, desc="Reading Dataset.jsonl", total=max_lines)):
            if i >= max_lines:
                break
            try:
                article = json.loads(line)
                for mesh in article.get('mesh', []):
                    code = mesh.get('ui', '')
                    name = mesh.get('name', '')
                    if code and name and code not in mesh_names:
                        mesh_names[code] = name
            except json.JSONDecodeError:
                continue
    
    print(f"Loaded {len(mesh_names)} unique MeSH codes")
    return mesh_names


def load_embeddings(year):
    """Load embeddings for a specific year"""
    filepath = OUTPUT_DIR / f"embeddings_{year}.txt"
    
    codes = []
    embeddings = []
    
    with open(filepath, 'r') as f:
        # First line contains dimensions
        first_line = f.readline().strip().split()
        n_codes, n_dims = int(first_line[0]), int(first_line[1])
        
        # Read each embedding
        for line in f:
            parts = line.strip().split()
            if len(parts) > 1:
                codes.append(parts[0])
                embeddings.append([float(x) for x in parts[1:]])
    
    return np.array(codes), np.array(embeddings)


def reduce_dimensions(embeddings, method='pca+tsne'):
    """Reduce embeddings to 2D using PCA+TSNE or PCA only"""
    if method == 'pca+tsne':
        # First reduce to 50D with PCA (fast), then to 2D with t-SNE
        pca = PCA(n_components=min(50, len(embeddings)-1), random_state=RANDOM_STATE)
        embeddings_pca = pca.fit_transform(embeddings)
        
        tsne = TSNE(n_components=2, random_state=RANDOM_STATE, 
                   perplexity=min(PERPLEXITY, len(embeddings)-1),
                   max_iter=300, verbose=0, n_jobs=-1)
        result = tsne.fit_transform(embeddings_pca)
        return result
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=RANDOM_STATE, 
                      perplexity=min(PERPLEXITY, len(embeddings)-1),
                      max_iter=300, verbose=0)
        return reducer.fit_transform(embeddings)
    else:  # PCA only
        reducer = PCA(n_components=2, random_state=RANDOM_STATE)
        return reducer.fit_transform(embeddings)


def procrustes_align(X, Y):
    """
    Align X to Y using Procrustes analysis (rotation + reflection).
    
    Args:
        X: Source point cloud (n_points, 2)
        Y: Target point cloud (n_points, 2)
    
    Returns:
        Aligned X that best matches Y
    """
    # Center both point clouds
    X_centered = X - X.mean(axis=0)
    Y_centered = Y - Y.mean(axis=0)
    
    # Compute optimal rotation matrix using SVD
    # R minimizes ||X @ R - Y||^2
    R, _ = orthogonal_procrustes(X_centered, Y_centered)
    
    # Apply rotation and translation
    X_aligned = X_centered @ R + Y.mean(axis=0)
    
    return X_aligned


def cluster_embeddings(embeddings, n_clusters=N_CLUSTERS):
    """Cluster embeddings using K-Means"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    return labels, kmeans


def load_cluster_names(filepath="cluster_names.json"):
    """Load cluster names from JSON file if available"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Convert string keys to integers
            cluster_names = {int(k): v for k, v in data["cluster_names"].items()}
            print(f"Loaded cluster names from {filepath}")
            return cluster_names
    except FileNotFoundError:
        print(f"Warning: {filepath} not found. Using default naming.")
        return None


def get_cluster_names(codes, labels, mesh_names, n_clusters=N_CLUSTERS):
    """Assign names to clusters based on most common terms (fallback method)"""
    cluster_names = {}
    
    for cluster_id in range(n_clusters):
        # Get codes in this cluster
        cluster_codes = codes[labels == cluster_id]
        
        # Get names for these codes
        cluster_terms = []
        for code in cluster_codes[:50]:  # Sample first 50
            if code in mesh_names:
                name = mesh_names[code]
                # Extract key terms (first 2 words)
                terms = name.split(',')[0].split()[:2]
                cluster_terms.extend(terms)
        
        # Find most common term
        if cluster_terms:
            from collections import Counter
            most_common = Counter(cluster_terms).most_common(3)
            cluster_name = ' / '.join([term for term, _ in most_common])
        else:
            cluster_name = f"Area {cluster_id + 1}"
        
        cluster_names[cluster_id] = cluster_name
    
    return cluster_names


def plot_embeddings_year(year, codes, embeddings_2d, labels, cluster_names, 
                         mesh_names, save_path=None):
    """Create a 2D plot for a single year"""
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor(FACECOLOR)
    ax.set_facecolor(FACECOLOR)
    
    # Plot each cluster
    for cluster_id in range(N_CLUSTERS):
        mask = labels == cluster_id
        if DARKMODE:
            ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                      c=COLORS[cluster_id], label=cluster_names[cluster_id],
                      alpha=0.6, s=30, edgecolors='white', linewidth=0.5)
        else:
            ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                      c=COLORS[cluster_id], label=cluster_names[cluster_id],
                      alpha=0.6, s=30, edgecolors='black', linewidth=0.3)
    
    # Add title and legend
    ax.set_title(f'MeSH Code Embeddings - Year {year}\n'
                f'Dimensionality Reduction: PCA+t-SNE, Clustering: K-Means (k={N_CLUSTERS})',
                fontsize=16, fontweight='bold', pad=20, color=TITLE_COLOR)
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12, color=TITLE_COLOR)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12, color=TITLE_COLOR)
    
    # Legend
    legend = ax.legend(loc='upper right', fontsize=10, framealpha=0.9, 
                      edgecolor=EDGE_COLOR, title='Technological Areas', title_fontsize=11)
    legend.get_frame().set_facecolor(FACECOLOR)
    if DARKMODE:
        legend.get_frame().set_edgecolor('white')
        for text in legend.get_texts():
            text.set_color('white')
        legend.get_title().set_color('white')
    else:
        legend.get_frame().set_edgecolor('black')
        for text in legend.get_texts():
            text.set_color('black')
        legend.get_title().set_color('black')
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, color=GRID_COLOR)
    
    # Tick colors
    ax.tick_params(colors=TITLE_COLOR)
    
    # Remove axis ticks for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=FACECOLOR)
        plt.close()
    else:
        plt.show()


def create_animation(image_files, output_path='embeddings_evolution.gif', fps=0.5):
    """Create animated GIF from image files"""
    print(f"\nCreating animated GIF (FPS: {fps})...")
    images = []
    for filepath in tqdm(image_files, desc="Loading frames"):
        images.append(imageio.imread(filepath))
    
    imageio.mimsave(output_path, images, fps=fps, loop=0)
    print(f"Animation saved to {output_path}")


def main():
    print("="*70)
    print("MeSH Code Embedding Visualization".center(70))
    print("="*70)
    print()
    
    # Load MeSH names
    mesh_names = load_mesh_names()
    print()
    
    # Load predefined cluster names if available
    predefined_names = load_cluster_names()
    
    # Create output directory for plots
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    
    image_files = []
    reference_embeddings_2d = None
    
    # First pass: Load all embeddings and reduce dimensions
    print(f"\nProcessing embeddings for all years...")
    print(f"Reference year for alignment: {REFERENCE_YEAR}")
    print()
    
    year_data = {}
    
    for year in tqdm(YEARS, desc="Loading & reducing dimensions"):
        # Load embeddings
        codes, embeddings = load_embeddings(year)
        
        # Reduce dimensions (PCA first, then t-SNE for speed)
        embeddings_2d = reduce_dimensions(embeddings, method='pca+tsne')
        
        # Store reference year embeddings for Procrustes alignment
        if year == REFERENCE_YEAR:
            reference_embeddings_2d = embeddings_2d.copy()
            print(f"\nStored reference embeddings from year {REFERENCE_YEAR}")
        
        # Cluster
        labels, kmeans = cluster_embeddings(embeddings)
        
        # Get cluster names (use reference year or predefined names)
        if year == REFERENCE_YEAR:
            if predefined_names is not None:
                reference_cluster_names = predefined_names
            else:
                reference_cluster_names = get_cluster_names(codes, labels, mesh_names)
        
        year_data[year] = {
            'codes': codes,
            'embeddings_2d': embeddings_2d,
            'labels': labels,
        }
    
    # Apply Procrustes alignment to all years
    print("\nApplying Procrustes alignment to stabilize orientations...")
    for year in tqdm(YEARS, desc="Aligning embeddings"):
        if year != REFERENCE_YEAR:
            # Align to reference year
            aligned = procrustes_align(year_data[year]['embeddings_2d'], reference_embeddings_2d)
            year_data[year]['embeddings_2d'] = aligned
    
    print("\nGenerating plots for each year...")
    print()
    
    # Second pass: Generate plots
    for year in tqdm(YEARS, desc="Creating plots"):
        data = year_data[year]
        
        # Use reference cluster names for consistency
        cluster_names = reference_cluster_names
        
        # Plot
        save_path = plots_dir / f"embeddings_{year}.png"
        plot_embeddings_year(
            year, 
            data['codes'],
            data['embeddings_2d'],
            data['labels'],
            cluster_names,
            mesh_names,
            save_path=save_path
        )
        image_files.append(save_path)
    
    print()
    print("Cluster Names (Technological Areas):")
    print("-" * 50)
    for cluster_id, name in reference_cluster_names.items():
        print(f"  Cluster {cluster_id}: {name}")
    print()
    
    # Sample codes from each cluster (using 2015 as example)
    print("Sample MeSH codes from each cluster (Year 2015):")
    print("-" * 50)
    year_2015_data = year_data[2015]
    for cluster_id in range(N_CLUSTERS):
        mask = year_2015_data['labels'] == cluster_id
        sample_codes = year_2015_data['codes'][mask][:5]
        print(f"\n  {reference_cluster_names[cluster_id]}:")
        for code in sample_codes:
            name = mesh_names.get(code, "Unknown")
            print(f"    {code}: {name}")
    print()
    
    # Create animation (fps=0.5 means 1 frame every 2 seconds)
    create_animation(image_files, 'embeddings_evolution.gif', fps=0.6)
    
    print()
    print("="*70)
    print("Visualization Complete!".center(70))
    print("="*70)
    print(f"\nIndividual plots saved in: {plots_dir}/")
    print(f"Animation saved as: embeddings_evolution.gif")
    print()


if __name__ == "__main__":
    main()
