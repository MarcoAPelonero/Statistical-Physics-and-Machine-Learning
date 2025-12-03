// Define a notion of graph, that has to be compatible for both the Ising model and the Voter model
#pragma once
#include <vector>
#include <utility> // for std::pair
#include <cstddef> // for std::size_t
#include <cstdlib> //  for std::rand
#include <cmath>   // for std::exp
#include <fstream> // for std::ofstream

struct Edge {
    std::size_t from;
    std::size_t to;
    double weight; // weight of the edge, can represent interaction strength or influence
};

class Graph {
public:
    // Make 3 types of constructors, one generates a fully connected graph, one generates a 
    // random graph with probability of connection p parameter
    // one makes the "small world" network, so it starts with a network where everyone is connecected to n first neighbors
    // and then rewires edges with probability p
    // For each constructor, initialize also a vector of node states (for Ising or Voter model)
    // each time just picking randomly +1 or -1
    Graph(int num_nodes); // Fully connected graph
    Graph(double length); // Lattice graph
    Graph(int num_nodes, double p); // Random graph
    Graph(int num_nodes, int n, double p); // Small world network

    void make_graph_undirected(); // Function to make the graph undirected (implementation not shown)

    void plot() const; // Function to plot the graph (implementation not shown)
    void save_edges_to_file(std::ofstream& file) const;

    void update_node_state_voter(std::size_t node);
    void update_node_state_ising(std::size_t node, double temperature);

    void update_graph_voter(); // Update the entire graph for Voter model (implementation not shown)
    void update_graph_ising(double temperature); // Update the entire graph for Ising model (

    void polarize(bool state); // Set all node states to +1 or -1
    void droplet(double radius, bool state); // Create a droplet of +1 or -1 spins in the center of the lattice

private:
    int num_nodes;
    std::vector<std::vector<std::pair<std::size_t, double>>> adjacency_list;
    std::vector<int> nodes_state;

};