"""
Manually curated cluster names based on analysis of MeSH code distribution.
Edit this file to customize cluster names for your visualization.
"""

import json

# Analyze the cluster content and provide better names
# Based on the sample codes and keywords from name_clusters.py output

IMPROVED_CLUSTER_NAMES = {
    0: "Physiology / Acoustic / Neurology",
    1: "Biochemistry / Pharmacology",
    2: "Clinical Medicine / Surgery",
    3: "Oncology / Behavioral Sciences", 
    4: "Orthopedics / Dentistry",
    5: "Public Health / Medical Ethics",
    6: "Molecular Biology / Genetics",
    7: "Veterinary / Immunology"
}

def save_improved_names():
    """Save improved cluster names to JSON file"""
    
    # Load existing analysis data if available
    try:
        with open("cluster_names.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {
            "reference_year": 2015,
            "n_clusters": 8,
            "cluster_analyses": {}
        }
    
    # Update with improved names
    data["cluster_names"] = IMPROVED_CLUSTER_NAMES
    
    # Save back
    with open("cluster_names.json", 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    print("Improved cluster names saved to cluster_names.json:")
    print()
    for cluster_id, name in IMPROVED_CLUSTER_NAMES.items():
        print(f"  Cluster {cluster_id}: {name}")
    print()
    print("Run visualize_embeddings.py again to use these names.")


if __name__ == "__main__":
    save_improved_names()
