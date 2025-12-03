#include "graph.hpp"
#include <iostream>
#include <vector>
#include <fstream>

int main() {
    // Generate a lattice graph, and switch all the nodes to state 1
    Graph g(250.0);
    g.make_graph_undirected();
    g.polarize(true); // Set all nodes to +1

    g.droplet(75.0, false); // Create a droplet of -1 spins in the center with radius 20

    // Open file to save lattice states
    std::ofstream outfile("lattice_voter_diffusion.txt");
    
    // Save initial state (step 0)
    outfile << "# Step 0\n";
    g.save_edges_to_file(outfile);
    
    // Run simulation for 10000 steps
    for (int step = 1; step <= 20000; ++step) {
        g.update_graph_voter();
        
        // Save state every step
        if (step % 300 == 0) {
            outfile << "# Step " << step << "\n";
            g.save_edges_to_file(outfile);
        }
        
        // Print progress every 1000 steps
        if (step % 1000 == 0) {
            std::cout << "Completed step " << step << std::endl;
        }
    }
    
    outfile.close();
    std::cout << "Simulation complete! Data saved to lattice_voter_diffusion.txt" << std::endl;
    
    return 0;
}