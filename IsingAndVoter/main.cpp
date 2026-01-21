#include "graph.hpp"
#include <fstream>
#include <iostream>

int main() {
    Graph fully_connected_graph(50);
    // fully_connected_graph.plot();

    Graph lattice_graph(7.0);
    // lattice_graph.plot();

    Graph random_graph(50, 0.3);
    // random_graph.plot();

    Graph small_world_graph(50, 4, 0.1);
    // small_world_graph.plot();

    fully_connected_graph.update_graph_voter();
    random_graph.update_graph_voter();
    small_world_graph.update_graph_voter();

    // Perform simulation steps of the Ising model at temperature T=0 and voter for a lattice
    // Save only the final state

    double temperature = 0.0;
    int total_steps = 2000;

    Graph lattice_graph_ising(50.0);
    lattice_graph_ising.make_graph_undirected();

    for (int step = 0; step < total_steps; ++step) {
        lattice_graph_ising.update_graph_ising(temperature);
        if ((step + 1) % 100 == 0) {
            std::cout << "Done step " << (step + 1) << " out of " << total_steps << "\n";
        }
    }

    // Save only the final state
    std::ofstream file("lattice_ising.txt");
    file << "# Step " << total_steps << "\n";
    lattice_graph_ising.save_nodes_to_file(file);
    file.close();

    std::cout << "Simulation completed for Ising. Final state saved to lattice_ising.txt\n";

    // Reinitialize the lattice graph for the voter model with random initial states
    Graph lattice_graph_voter(500.0);
    lattice_graph_voter.make_graph_undirected();

    total_steps = 40000;

    for (int step = 0; step < total_steps; ++step) {
        lattice_graph_voter.update_graph_voter(lattice_graph_voter.get_num_nodes());
        if ((step + 1) % 100 == 0) {
            std::cout << "Done step " << (step + 1) << " out of " << total_steps << "\n";
        }
    }

    // Save only the final state
    std::ofstream voter_file("lattice_voter.txt");
    voter_file << "# Step " << total_steps << "\n";
    lattice_graph_voter.save_nodes_to_file(voter_file);
    voter_file.close();

    std::cout << "Simulation completed for Voter. Final state saved to lattice_voter.txt\n";

    return 0;
}