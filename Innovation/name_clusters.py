"""
Analyze clusters and generate meaningful names based on MeSH code content.
This script samples codes from each cluster and provides semantic analysis.
"""

import json
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Configuration
OUTPUT_DIR = Path("output")
REFERENCE_YEAR = 2015  # Use middle year for reference
N_CLUSTERS = 8
RANDOM_STATE = 42
SAMPLE_SIZE = 100  # Sample this many codes per cluster


def load_mesh_names(dataset_path="Dataset.jsonl", max_lines=100000):
    """Load MeSH code to name mapping"""
    print("Loading MeSH codes from Dataset.jsonl...")
    mesh_names = {}
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
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


def cluster_embeddings(embeddings, n_clusters=N_CLUSTERS):
    """Cluster embeddings using K-Means"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    return labels


def analyze_cluster(codes, mesh_names, sample_size=SAMPLE_SIZE):
    """Analyze a cluster's MeSH codes to understand its theme"""
    # Get names for sampled codes
    names = []
    for code in codes[:sample_size]:
        if code in mesh_names:
            names.append(mesh_names[code])
    
    if not names:
        return {"raw_names": [], "keywords": [], "summary": "Unknown"}
    
    # Extract key terms from names
    all_words = []
    for name in names:
        # Split on comma and slash, take meaningful words
        words = name.replace('/', ' ').replace(',', ' ').split()
        # Filter out common short words
        words = [w for w in words if len(w) > 3 and w.lower() not in 
                ['with', 'from', 'that', 'this', 'have', 'been', 'were']]
        all_words.extend(words)
    
    # Get most common terms
    word_counts = Counter(all_words)
    top_keywords = [word for word, count in word_counts.most_common(10)]
    
    return {
        "raw_names": names[:20],  # Store first 20 for reference
        "keywords": top_keywords,
        "code_count": len(codes)
    }


def suggest_cluster_name(analysis):
    """Suggest a meaningful name based on cluster analysis"""
    keywords = analysis["keywords"]
    raw_names = analysis["raw_names"]
    
    if not keywords:
        return "Unknown Area"
    
    # Group by semantic similarity (manual mapping of common medical domains)
    # This is a simple heuristic - you can improve it
    
    # Analyze the top keywords and raw names to infer the domain
    keywords_str = " ".join(keywords[:5]).lower()
    names_str = " ".join(raw_names[:10]).lower()
    
    combined = keywords_str + " " + names_str
    
    # Simple keyword-based categorization
    if any(term in combined for term in ['disease', 'syndrome', 'disorder', 'pathology', 'infection']):
        if any(term in combined for term in ['neuro', 'brain', 'cognitive', 'mental', 'psychiatric']):
            return "Neurological / Psychiatric Disorders"
        elif any(term in combined for term in ['cardiovascular', 'heart', 'cardiac', 'vascular']):
            return "Cardiovascular Diseases"
        elif any(term in combined for term in ['cancer', 'tumor', 'neoplasm', 'carcinoma']):
            return "Oncological Conditions"
        else:
            return " / ".join(keywords[:3])
    
    elif any(term in combined for term in ['therapy', 'treatment', 'drug', 'pharmaceutical', 'medicine']):
        return "Therapeutics / Pharmacology"
    
    elif any(term in combined for term in ['genetic', 'gene', 'molecular', 'protein', 'cell']):
        return "Molecular / Genetic Research"
    
    elif any(term in combined for term in ['surgery', 'surgical', 'procedure', 'operative']):
        return "Surgical Procedures"
    
    elif any(term in combined for term in ['diagnostic', 'imaging', 'test', 'screening']):
        return "Diagnostics / Imaging"
    
    else:
        # Fall back to top 3 keywords
        return " / ".join(keywords[:3])


def main():
    print("="*70)
    print("Cluster Analysis & Naming".center(70))
    print("="*70)
    print()
    
    # Load MeSH names
    mesh_names = load_mesh_names()
    print()
    
    # Load reference year embeddings
    print(f"Loading embeddings for reference year {REFERENCE_YEAR}...")
    codes, embeddings = load_embeddings(REFERENCE_YEAR)
    print(f"Loaded {len(codes)} codes with {embeddings.shape[1]}D embeddings")
    print()
    
    # Cluster
    print(f"Clustering into {N_CLUSTERS} clusters...")
    labels = cluster_embeddings(embeddings)
    print()
    
    # Analyze each cluster
    print("Analyzing clusters...")
    print("="*70)
    print()
    
    cluster_analyses = {}
    suggested_names = {}
    
    for cluster_id in range(N_CLUSTERS):
        print(f"Cluster {cluster_id}:")
        print("-" * 70)
        
        # Get codes in this cluster
        mask = labels == cluster_id
        cluster_codes = codes[mask]
        
        # Analyze
        analysis = analyze_cluster(cluster_codes, mesh_names)
        cluster_analyses[cluster_id] = analysis
        
        # Suggest name
        suggested_name = suggest_cluster_name(analysis)
        suggested_names[cluster_id] = suggested_name
        
        # Print analysis
        print(f"  Size: {analysis['code_count']} codes")
        print(f"  Suggested Name: {suggested_name}")
        print(f"  Top Keywords: {', '.join(analysis['keywords'][:8])}")
        print(f"\n  Sample MeSH Terms:")
        for i, name in enumerate(analysis['raw_names'][:8], 1):
            print(f"    {i}. {name}")
        print()
    
    # Save suggested names to file
    output_file = "cluster_names.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "reference_year": REFERENCE_YEAR,
            "n_clusters": N_CLUSTERS,
            "cluster_names": suggested_names,
            "cluster_analyses": {k: {"keywords": v["keywords"], "code_count": v["code_count"]} 
                                for k, v in cluster_analyses.items()}
        }, f, indent=2)
    
    print("="*70)
    print(f"Analysis saved to {output_file}")
    print("="*70)
    print()
    print("Recommended cluster names:")
    for cluster_id, name in suggested_names.items():
        print(f"  Cluster {cluster_id}: {name}")
    print()
    print("Review these names and edit cluster_names.json if needed.")
    print("The visualization script will read from this file.")
    print()


if __name__ == "__main__":
    main()
