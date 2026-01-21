#include "graph.hpp"
#include <iostream>
#include <vector>
#include <ctime>
#include <fstream>
#include <limits>
#include <cmath>

int main() {
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    const double length = 25.0; // lattice side length
    const int repeats = 500;
    // 5 target densities between 0 and 1 (we aim for fraction of +1 spins)
    const std::vector<double> targets = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};

    Graph dummyGraph(length, 0.1);
    std::cout << "Graph with " << dummyGraph.get_num_nodes() << " nodes created for length " << length << std::endl;

    std::ofstream out("consensus_sweep.csv");
    out << "target,frac_plus,repeats,total_consensus,plus_count,minus_count,avg_steps\n";

    for (double target : targets) {
        int total_consensus = 0;
        int plus_count = 0;
        int minus_count = 0;
        long long steps_sum = 0;
        std::cout << "Running target density: " << target << std::endl;

        for (int rep = 0; rep < repeats; ++rep) {
            Graph g(length, target); // fresh graph with desired density

            int steps = 0;
            bool consensus = false;
            const int max_iters = 200000; // safety cap

            while (!consensus && steps < max_iters) {
                g.update_graph_voter(g.get_num_nodes());
                // g.update_graph_ising(0.0);
                steps++;
                consensus = g.consensus_reached();
            }

            // Final check in case loop ended due to cap
            if (!consensus) consensus = g.consensus_reached();

            if (consensus) {
                total_consensus++;
                double d = g.density();
                if (d > 0.5) ++plus_count; else ++minus_count;
                steps_sum += steps;
                // std::cout << "Run " << rep << " for target " << target << " reached consensus in " << steps << " steps.\n";
            }
            else {
                std::cout << "Run " << rep << " for target " << target << " did not reach consensus in " << max_iters << " steps.\n";
            }
        }

        double avg_steps = total_consensus > 0 ? static_cast<double>(steps_sum) / total_consensus : 0.0;
        out << target << "," 
            << (static_cast<double>(plus_count) / repeats) << "," 
            << repeats << "," 
            << total_consensus << "," 
            << plus_count << "," 
            << minus_count << "," 
            << avg_steps << "\n";

        // Also print a concise summary to stdout
        std::cout << "target=" << target << "Exit:" <<static_cast<double>(total_consensus)/repeats*100.0 << "% consensus (" << plus_count << " +1, " << minus_count << " -1), avg steps: " << avg_steps << std::endl;
    }

    out.close();
    std::cout << "Results written to consensus_sweep.csv" << std::endl;
    return 0;
}