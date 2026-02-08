#include "graph.hpp"
#include <iostream>
#include <vector>
#include <ctime>
#include <fstream>
#include <limits>
#include <cmath>

int main() {
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    const double length = 50.0; // lattice side length
    const int repeats = 100;
    // 5 target densities between 0 and 1 (we aim for fraction of +1 spins)
    const std::vector<double> targets = {0.1, 0.3, 0.5, 0.7, 0.9};

    std::ofstream out("consensus_sweep.csv");
    out << "target_density,achieved_density,repeats,total_consensus,plus_count,minus_count,avg_steps_when_consensus\n";

    for (double target : targets) {
        int total_consensus = 0;
        int plus_count = 0;
        int minus_count = 0;
        long long steps_sum = 0;

        #pragma omp parallel  for
        for (int rep = 0; rep < repeats; ++rep) {
            Graph g(length, target); // fresh graph with desired density

            int steps = 0;
            bool consensus = false;
            const int max_iters = 200000; // safety cap

            while (!consensus && steps < max_iters) {
                g.update_graph_ising(g.get_num_nodes());
                steps++;
                if (steps % 100 == 0) {
                    consensus = g.consensus_reached();
                }
            }

            // Final check in case loop ended due to cap
            if (!consensus) consensus = g.consensus_reached();

            if (consensus) {
                total_consensus++;
                double d = g.density();
                if (d > 0.5) ++plus_count; else ++minus_count;
                steps_sum += steps;
            }
        }

        double avg_steps = total_consensus > 0 ? static_cast<double>(steps_sum) / total_consensus : 0.0;
        out << target << "," << "," << repeats << "," << total_consensus << "," << plus_count << "," << minus_count << "," << avg_steps << "\n";

        // Also print a concise summary to stdout
        std::cout << "target=" << target << static_cast<double>(total_consensus)/repeats*100.0 << "% consensus (" << plus_count << " +1, " << minus_count << " -1), avg steps: " << avg_steps << std::endl;
    }

    out.close();
    std::cout << "Results written to consensus_sweep.csv" << std::endl;
    return 0;
}