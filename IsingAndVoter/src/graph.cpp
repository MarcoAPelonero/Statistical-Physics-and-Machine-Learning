#include "graph.hpp"
#include <fstream>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

Graph::Graph(int num_nodes) : num_nodes(num_nodes) {
    adjacency_list.resize(num_nodes);
    for (int i = 0; i < num_nodes; ++i) {
        for (int j = 0; j < num_nodes; ++j) {
            if (i != j) {
                adjacency_list[static_cast<std::size_t>(i)].emplace_back(static_cast<std::size_t>(j), 1.0); // Fully connected with weight 1.0
            }
        }
    }
    nodes_state.resize(num_nodes);
    for (int i = 0; i < num_nodes; ++i) {
        nodes_state[i] = (rand() % 2) * 2 - 1; // Randomly assign +1 or -1
    }
}

Graph::Graph(double length) {
    num_nodes = static_cast<int>(length * length);
    adjacency_list.resize(num_nodes);
    for (int i = 0; i < static_cast<int>(length); ++i) {
        for (int j = 0; j < static_cast<int>(length); ++j) {
            int node = i * static_cast<int>(length) + j;
            // Connect to right neighbor
            if (j < static_cast<int>(length) - 1) {
                int right_neighbor = i * static_cast<int>(length) + (j + 1);
                adjacency_list[static_cast<std::size_t>(node)].emplace_back(static_cast<std::size_t>(right_neighbor), 1.0);
            }
            // Connect to bottom neighbor
            if (i < static_cast<int>(length) - 1) {
                int bottom_neighbor = (i + 1) * static_cast<int>(length) + j;
                adjacency_list[static_cast<std::size_t>(node)].emplace_back(static_cast<std::size_t>(bottom_neighbor), 1.0);
            }
        }
    }
    nodes_state.resize(num_nodes);
    for (int i = 0; i < num_nodes; ++i) {
        nodes_state[i] = (rand() % 2) * 2 - 1; // Randomly assign +1 or -1
    }
}

Graph::Graph(double length, double p) : num_nodes(static_cast<int>(length * length)) {
    // Start by making a fully connected graph
    adjacency_list.resize(num_nodes);
    for (int i = 0; i < num_nodes; ++i) {
        for (int j = 0; j < num_nodes; ++j) {
            if (i != j) {
                adjacency_list[static_cast<std::size_t>(i)].emplace_back(static_cast<std::size_t>(j), 1.0); // Fully connected with weight 1.0
            }
        }
    }
    // Make it bidirectional
    make_graph_undirected();
    nodes_state.resize(num_nodes);
    for (int i = 0; i < num_nodes; ++i) {
        double u = static_cast<double>(rand()) / RAND_MAX;
        if (u < p) {
            nodes_state[i] = 1; // +1 state
        } else {
            nodes_state[i] = -1; // -1 state
        }
    }
}

Graph::Graph(int num_nodes, double p) : num_nodes(num_nodes) {
    adjacency_list.resize(num_nodes);
    for (int i = 0; i < num_nodes; ++i) {
        for (int j = 0; j < num_nodes; ++j) {
            if (i != j) {
                if ((static_cast<double>(rand()) / RAND_MAX) < p) {
                    adjacency_list[static_cast<std::size_t>(i)].emplace_back(static_cast<std::size_t>(j), 1.0); // Random connection with weight 1.0
                }
            }
        }
    }
    nodes_state.resize(num_nodes);
    for (int i = 0; i < num_nodes; ++i) {
        nodes_state[i] = (rand() % 2) * 2 - 1; // Randomly assign +1 or -1
    }
}

Graph::Graph(int num_nodes, int n, double p) : num_nodes(num_nodes) {
    adjacency_list.resize(num_nodes);
    // Step 1: Create a ring lattice where each node is connected to n neighbors
    for (int i = 0; i < num_nodes; ++i) {
        for (int j = 1; j <= n / 2; ++j) {
            std::size_t neighbor1 = static_cast<std::size_t>((i + j) % num_nodes);
            std::size_t neighbor2 = static_cast<std::size_t>((i + num_nodes - j) % num_nodes);
            adjacency_list[static_cast<std::size_t>(i)].emplace_back(neighbor1, 1.0);
            adjacency_list[static_cast<std::size_t>(i)].emplace_back(neighbor2, 1.0);
        }
    }
    // Step 2: Rewire edges with probability p
    for (int i = 0; i < num_nodes; ++i) {
        for (auto& edge : adjacency_list[static_cast<std::size_t>(i)]) {
            if ((static_cast<double>(rand()) / RAND_MAX) < p) {
                std::size_t new_target;
                do {
                    new_target = static_cast<std::size_t>(rand() % num_nodes);
                } while (new_target == static_cast<std::size_t>(i)); // Avoid self-loops
                edge.first = new_target; // Rewire to new target
            }
        }
    }
    nodes_state.resize(num_nodes);
    for (int i = 0; i < num_nodes; ++i) {
        nodes_state[i] = (rand() % 2) * 2 - 1; // Randomly assign +1 or -1
    }
}

void Graph::make_graph_undirected() {
    for (std::size_t i = 0; i < adjacency_list.size(); ++i) {
        for (const auto& edge : adjacency_list[i]) {
            std::size_t neighbor = edge.first;
            double weight = edge.second;
            // Check if the reverse edge already exists
            bool found = false;
            for (const auto& rev_edge : adjacency_list[neighbor]) {
                if (rev_edge.first == i) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                adjacency_list[neighbor].emplace_back(i, weight); // Add reverse edge
            }
        }
    }
}

void Graph::make_graph_fully_connected() {
    adjacency_list.clear();
    adjacency_list.resize(num_nodes);
    for (int i = 0; i < num_nodes; ++i) {
        for (int j = 0; j < num_nodes; ++j) {
            if (i != j) {
                adjacency_list[static_cast<std::size_t>(i)].emplace_back(static_cast<std::size_t>(j), 1.0); // Fully connected with weight 1.0
            }
        }
    }
}

void Graph::save_edges_to_file(std::ofstream& file) const {
    for (std::size_t i = 0; i < adjacency_list.size(); ++i) {
        for (const auto& edge : adjacency_list[i]) {
            file << i << " " << edge.first << " " << edge.second << " " << nodes_state[i] << "\n";
        }
    }
}

void Graph::save_nodes_to_file(std::ofstream& file) const {
    for (std::size_t i = 0; i < static_cast<std::size_t>(num_nodes); ++i) {
        file << "NODE " << i << " " << nodes_state[i] << "\n";
    }
}

void Graph::plot() const {
    std::ofstream file("edges.txt");
    for (std::size_t i = 0; i < adjacency_list.size(); ++i) {
        for (const auto& edge : adjacency_list[i]) {
            file << i << " " << edge.first << " " << edge.second << " " << nodes_state[i] << "\n";
        }
    }
    // Also write explicit node state lines so every node has a recorded state
    // Format: NODE <node_id> <state>
    for (std::size_t i = 0; i < static_cast<std::size_t>(num_nodes); ++i) {
        file << "NODE " << i << " " << nodes_state[i] << "\n";
    }
    file.close();

    std::vector<std::string> candidates = {"py -3", "python", "python3"};
    std::string python_cmd;
    for (const auto &c : candidates) {
        // Test whether this command can import sys. Redirect output to nul on Windows.
        std::string test = c + " -c \"import sys\" >nul 2>&1";
        int ret = system(test.c_str());
        if (ret == 0) {
            python_cmd = c;
            break;
        }
    }

    if (python_cmd.empty()) {
        std::cerr << "Python not found. To view plots, install Python 3 and the packages\n"
                  << "networkx and matplotlib, then run:\n"
                  << "  py -3 scripts\\plot_network.py edges.txt\n"
                  << "or install Python and ensure `python` is on your PATH." << std::endl;
        return;
    }

    std::string cmd = python_cmd + " scripts/plot_network.py edges.txt";
    int runret = system(cmd.c_str());
    if (runret != 0) {
        std::cerr << "Plotting script exited with code " << runret << "." << std::endl;
    }
}

void Graph::update_node_state_voter(std::size_t node) {
    // Select randomly one of the neighbors and copy its state
    // This method updates a single node (asynchronous update)
    if (adjacency_list[node].empty()) return; // No neighbors to copy from
    std::size_t random_index = static_cast<std::size_t>(rand() % adjacency_list[node].size());
    std::size_t neighbor = adjacency_list[node][random_index].first;
    nodes_state[node] = nodes_state[neighbor];
}

void Graph::update_node_state_ising(std::size_t node, double temperature) {
    // Compute local field h_i = sum_j J_ij * s_j
    // This method updates a single node (asynchronous update)
    double h = 0.0;
    for (const auto& edge : adjacency_list[node]) {
        h += edge.second * nodes_state[edge.first];
    }

    int s = nodes_state[node];
    double delta_E = 2.0 * s * h;   // energy change if spin is flipped

    // ---------- T = 0 case ----------
    if (temperature <= 0.0) {
        if (delta_E < 0.0) {
            nodes_state[node] = -s;   // flip only if energy decreases
        }
        return;
    }

    // ---------- T > 0: Metropolis ----------
    if (delta_E <= 0.0) {
        nodes_state[node] = -s;       // always flip if ΔE <= 0
    } else {
        double prob = std::exp(-delta_E / temperature);
        double u = static_cast<double>(std::rand()) / RAND_MAX;
        if (u < prob) {
            nodes_state[node] = -s;
        }
    }
}

void Graph::update_graph_voter(int updates_per_call) {
    if (num_nodes <= 0 || updates_per_call <= 0) {
        return;
    }

    for (int step = 0; step < updates_per_call; ++step) {
        std::size_t node = static_cast<std::size_t>(rand() % num_nodes);
        if (adjacency_list[node].empty()) {
            continue; // No neighbors, nothing to copy
        }
        std::size_t neighbor_index = static_cast<std::size_t>(rand() % adjacency_list[node].size());
        std::size_t neighbor = adjacency_list[node][neighbor_index].first;
        nodes_state[node] = nodes_state[neighbor];
    }
}

void Graph::update_graph_ising(double temperature) {
    // Synchronous update: all spins flip based on current state
    // Create a buffer copy to ensure we read from the OLD state while writing to NEW state
    std::vector<int> new_state = nodes_state; // Copy current state as buffer
    
    for (std::size_t i = 0; i < adjacency_list.size(); ++i) {
        // Compute local field from CURRENT configuration (read from nodes_state, NOT new_state)
        double h = 0.0;
        for (const auto& edge : adjacency_list[i]) {
            // CRITICAL: Always read neighbor states from nodes_state (old state)
            h += edge.second * nodes_state[edge.first];
        }

        int s = nodes_state[i];  // Read current spin from old state
        double delta_E = 2.0 * s * h;

        // Decide whether to flip based on temperature (write to new_state buffer)
        if (temperature <= 0.0) {
            // T = 0: flip only if energy decreases
            if (delta_E < 0.0) {
                new_state[i] = -s;
            }
        } else {
            // T > 0: Metropolis algorithm
            if (delta_E <= 0.0) {
                new_state[i] = -s;
            } else {
                double prob = std::exp(-delta_E / temperature);
                double u = static_cast<double>(std::rand()) / RAND_MAX;
                if (u < prob) {
                    new_state[i] = -s;
                }
            }
        }
    }
    
    // Apply all updates simultaneously by swapping the entire state vector
    nodes_state = new_state;
}

void Graph::polarize(bool state) {
    int spin = state ? 1 : -1;
    for (int i = 0; i < num_nodes; ++i) {
        nodes_state[i] = spin;
    }
}

void Graph::droplet(double radius, bool state) {
    int spin = state ? 1 : -1;
    // Assuming the graph is a lattice, find the center
    int length = static_cast<int>(std::sqrt(num_nodes));
    int center_x = length / 2;
    int center_y = length / 2;

    for (int i = 0; i < length; ++i) {
        for (int j = 0; j < length; ++j) {
            int node = i * length + j;
            double dist = std::sqrt((i - center_x) * (i - center_x) + (j - center_y) * (j - center_y));
            if (dist <= radius) {
                nodes_state[node] = spin;
            }
        }
    }
}

double Graph::density() const {
    int count_plus_one = 0;
    for (int state : nodes_state) {
        if (state == 1) {
            ++count_plus_one;
        }
    }
    return static_cast<double>(count_plus_one) / num_nodes;
}

bool Graph::consensus_reached() const {
    if (nodes_state.empty()) return true;
    int first_state = nodes_state[0];
    for (int state : nodes_state) {
        if (state != first_state) {
            return false;
        }
    }
    return true;
}