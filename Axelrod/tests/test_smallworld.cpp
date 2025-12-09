#include <iostream>
#include <vector>
#include <algorithm>
#include <queue>
#include <numeric>
#include "graph.hpp"

bool check_no_self_loops(const StrogatzGraph& g) {
    const auto& adj = g.adjacency();
    int N = g.N();
    for (int i = 0; i < N; ++i) {
        for (int v : adj[i]) {
            if (v == i) {
                std::cerr << "Self-loop at node " << i << "\n";
                return false;
            }
        }
    }
    return true;
}

bool check_symmetry(const StrogatzGraph& g) {
    const auto& adj = g.adjacency();
    int N = g.N();
    for (int i = 0; i < N; ++i) {
        for (int v : adj[i]) {
            auto& row = adj[v];
            if (std::find(row.begin(), row.end(), i) == row.end()) {
                std::cerr << "Asymmetry: " << i << " -> " << v
                          << " exists, but not " << v << " -> " << i << "\n";
                return false;
            }
        }
    }
    return true;
}

bool check_no_duplicates(const StrogatzGraph& g) {
    const auto& adj = g.adjacency();
    int N = g.N();
    for (int i = 0; i < N; ++i) {
        std::vector<int> tmp = adj[i];
        std::sort(tmp.begin(), tmp.end());
        if (std::adjacent_find(tmp.begin(), tmp.end()) != tmp.end()) {
            std::cerr << "Duplicate neighbors in node " << i << "\n";
            return false;
        }
    }
    return true;
}

void print_degree_stats(const StrogatzGraph& g) {
    const auto& adj = g.adjacency();
    int N = g.N();
    std::vector<int> deg(N);
    for (int i = 0; i < N; ++i) {
        deg[i] = (int)adj[i].size();
    }
    int min_deg = *std::min_element(deg.begin(), deg.end());
    int max_deg = *std::max_element(deg.begin(), deg.end());
    double avg_deg = std::accumulate(deg.begin(), deg.end(), 0.0) / N;
    std::cout << "Degree stats: min=" << min_deg
              << " max=" << max_deg
              << " avg=" << avg_deg << "\n";
}

int main() {
    int num_nodes = 100;
    int neighbors_per_node = 4;
    double rewiring_prob = 0.1;
    int num_features = 5;
    int feature_dim = 3;

    StrogatzGraph graph(num_nodes, neighbors_per_node, rewiring_prob,
                        num_features, feature_dim);

    if (!check_no_self_loops(graph)) {
        std::cerr << "Graph has self-loops.\n";
        return 1;
    }

    if (!check_symmetry(graph)) {
        std::cerr << "Graph is not symmetric.\n";
        return 1;
    }

    if (!check_no_duplicates(graph)) {
        std::cerr << "Graph has duplicate edges.\n";
        return 1;
    }

    print_degree_stats(graph);
    
    std::cout << "All checks passed!\n";
    return 0;
}