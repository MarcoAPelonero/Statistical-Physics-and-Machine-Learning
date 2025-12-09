#pragma once    

#include "utils.hpp"
#include <vector>
#include <fstream>
#include <string>
#include <iostream>

struct couple {
    int node1;
    int node2;
};

class StrogatzGraph {
    private:
        int num_nodes;
        int neightbors_per_node;
        double rewiring_prob;
        int num_features;
        int feature_dim;
        std::vector<std::vector<int>> adj_list;
        std::vector<std::vector<int>> node_features;
        std::vector<int> culture_histogram;
        std::vector<double> culture_distribution;

    public:
        
        StrogatzGraph(int n, int k, double p, int f_num, int f_dim)
            : num_nodes(n), neightbors_per_node(k), rewiring_prob(p),
              num_features(f_num), feature_dim(f_dim) {
            generate_graph();
            generate_node_features();
        }

        void convert_to_2d_lattice();

        void generate_graph();
        void generate_node_features();

        couple select_nodes_for_interaction() const;

        const std::vector<std::vector<int>>& get_node_features() const;

        const std::vector<std::vector<int>>& adjacency() const { return adj_list; }
        int N() const { return num_nodes; }

        void interaction(int node1, int node2);
        void axelrod_interaction();
        void axelrod_step();

        void measure_culture_histogram();
        void normalize_culture_distribution();  

        std::vector<int> get_measure_culture_distribution() const { return culture_histogram; }
        std::vector<double> get_culture_distribution() const { return culture_distribution; }

        void save_culture_histogram(std::ofstream& outfile) const;
        void save_culture_distribution(std::ofstream& outfile) const;

        int count_distinct_cultures() const;
        int largest_culture_size() const;
        double average_similarity() const;
        double entropy_cultures() const;
        double fragmentation_index() const;
        double edge_homophily() const;
        double global_similarity() const;
};