#pragma once

#include "utils.hpp"
#include <fstream>
#include <string>
#include <vector>

struct couple {
    int node1;
    int node2;
};

class Graph {
  protected:
    int num_nodes;
    int num_features;
    int feature_dim;
    std::vector<std::vector<int>> adj_list;
    std::vector<std::vector<int>> node_features;
    std::vector<int> culture_histogram;
    std::vector<double> culture_distribution;

    Graph(int n, int f_num, int f_dim);
    virtual void generate_graph() = 0;
    void generate_node_features();
    couple select_nodes_for_interaction() const;
    void interaction(int node1, int node2);

  public:
    virtual ~Graph() = default;

    const std::vector<std::vector<int>>& adjacency() const { return adj_list; }
    const std::vector<std::vector<int>>& get_node_features() const { return node_features; }
    int N() const { return num_nodes; }

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

class StrogatzGraph : public Graph {
  public:
    StrogatzGraph(int n, int k, double p, int f_num, int f_dim);

  private:
    int neighbors_per_node;
    double rewiring_prob;
    void generate_graph() override;
};

class LatticeGraph : public Graph {
  public:
    LatticeGraph(int n, int radius, int f_num, int f_dim);

  private:
    int lattice_radius;
    void generate_graph() override;
    int lattice_side() const;
};
