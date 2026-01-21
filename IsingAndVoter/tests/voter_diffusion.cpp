#include "graph.hpp"
#include <iostream>
#include <vector>
#include <fstream>

int main() {
    // Generate a lattice graph, and switch all the nodes to state 1
    Graph g(20.0);
    g.make_graph_undirected();
    g.polarize(true); // Set all nodes to +1

    g.droplet(12.5, false); // Create a droplet of -1 spins in the center with radius 20
    

    // Open file to save lattice states
    std::ofstream outfile("lattice_voter_diffusion.txt");
    
    // Save initial state (step 0)
    outfile << "# Step 0\n";
    g.save_nodes_to_file(outfile);
    
    // Run simulation for 10000 steps
    for (int step = 1; step <= 200000; ++step) {
        g.update_graph_voter(g.get_num_nodes());
        
        // Save state every step
        if (step % 300 == 0) {
            outfile << "# Step " << step << "\n";
            g.save_nodes_to_file(outfile);
        }
        
        // Print progress every 1000 steps
        if (step % 10 == 0) {
            std::cout << "Completed step " << step << std::endl;
        }
    }
    
    outfile.close();
    std::cout << "Simulation complete! Data saved to lattice_voter_diffusion.txt" << std::endl;
    
    return 0;
}