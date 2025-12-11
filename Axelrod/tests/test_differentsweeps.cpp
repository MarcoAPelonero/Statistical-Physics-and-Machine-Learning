#include "graph.hpp"
#include "utils.hpp"
#include <iostream>
#include <vector>

std::vector<double> compute_distribution(StrogatzGraph g) {
    g.measure_culture_histogram();
    g.normalize_culture_distribution();
    return g.get_culture_distribution();
}

void print_vector(const std::vector<double>& vec) {
    for (double val : vec) {
        std::cout << val << " ";
    }
    std::cout << "\n";
}

void sweep_different_parameters(int num_nodes, int neighbors_per_node,
                                double rewiring_prob, int num_features,
                                int feature_dim, int num_interactions, bool verbose=false) {

    StrogatzGraph g(num_nodes, neighbors_per_node, rewiring_prob,
                    num_features, feature_dim);

    for (int it = 0; it < num_interactions; ++it) {
        if (it % 100 == 0) {
            std::vector<double> dist = compute_distribution(g);
            std::cout << "Interaction " << it << ", culture distribution: ";
            for (double val : dist) {
                std::cout << val << " ";
            }
            std::cout << "\n";
        }
        g.axelrod_interaction();
    }

    if (verbose) {
        const auto& features = g.get_node_features();
        for (size_t i = 0; i < features.size(); ++i) {
            std::cout << "Node " << i << ": ";
            for (size_t j = 0; j < features[i].size(); ++j) {
                std::cout << features[i][j] << " ";
            }
            std::cout << "\n";
        }
    }
}

int main () {
    // Make a series of default values and arrays of different values to sweep across, except for the number of nodes fixed at 500
    constexpr int def_num_nodes = 500;

    int def_neighbors_per_node = 10;
    double def_rewiring_prob = 0.1;
    int def_num_features = 5;
    int def_feature_dim = 3;

    std::vector<int> neighbors_per_node_values = {4, 10, 20, 50};
    std::vector<double> rewiring_prob_values = {0.0, 0.1, 0.3, 0.5, 0.7, 1.0};
    std::vector<int> num_features_values = {3, 5, 10, 20};
    std::vector<int> feature_dim_values = {2, 3, 5, 10};

    // For a sweep for now it means just creating a new graph and doing 3 updates with the Axelrod rule
    constexpr int num_interactions = 500;
    for (int k : neighbors_per_node_values) {
        sweep_different_parameters(def_num_nodes, k,
                                   def_rewiring_prob,
                                   def_num_features,
                                   def_feature_dim,
                                   num_interactions);
    }
    std::cout << "Completed sweep for neighbors per node.\n";
    for (double p : rewiring_prob_values) {
        sweep_different_parameters(def_num_nodes, def_neighbors_per_node,
                                   p,
                                   def_num_features,
                                   def_feature_dim,
                                   num_interactions);
    }
    std::cout << "Completed sweep for rewiring probability.\n";
    for (int f_num : num_features_values) {
        sweep_different_parameters(def_num_nodes, def_neighbors_per_node,
                                   def_rewiring_prob,
                                   f_num,
                                   def_feature_dim,
                                   num_interactions);
    }
    std::cout << "Completed sweep for number of features.\n";
    for (int f_dim : feature_dim_values) {
        sweep_different_parameters(def_num_nodes, def_neighbors_per_node,
                                   def_rewiring_prob,
                                   def_num_features,
                                   f_dim,
                                   num_interactions);
    }
    std::cout << "Completed sweep for feature dimension.\n";
    
    return 0;
}