#include "sandbox.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <utility>
#include <cstdlib>

// Experiment runner:
// - Runs N grain additions
// - Records only avalanches (steps with at least one topple)
// - Tracks and reports number of steps with no avalanches
// - Uses animationThreshold = 20

int main(int argc, char** argv) {
    // ---- configuration ----
    const int L = (argc > 1) ? std::atoi(argv[1]) : 100;          // lattice linear size
    const int total_steps = (argc > 2) ? std::atoi(argv[2]) : 30000;  // number of grain additions
    const int animationThreshold = 20;                            // start saving avalanche frames after this many topples
    const std::string framesFile = (argc > 3) ? argv[3] : "avalanche_frames.txt";
    int logEvery = 100;                                             // (not used currently)

    // ---- initialize sandbox ----
    Sandbox sandbox(L, animationThreshold, framesFile);

    // Save initial state (optional)
    {
        std::ofstream init_ofs("sandbox_initial.txt");
        sandbox.saveSandbox(init_ofs);
    }

    // ---- stats output ----
    std::ofstream csv("avalanches.csv");
    if (!csv) {
        std::cerr << "Error: cannot open avalanches.csv for writing.\n";
        return 1;
    }

    int no_avalanche_steps = 0;
    int avalanche_count = 0;

    // ---- simulation loop ----
    for (int step = 1; step <= total_steps; ++step) {
    Sandbox::AvalancheResult ar = sandbox.step(step);

        if (ar.topples > 0) {
            // first avalanche triggers writing header
            if (avalanche_count == 0) {
                csv << "# Total steps without avalanche will be added below\n";
                csv << "step,topples,affected,duration,max_distance,bbox_area,minx,miny,maxx,maxy\n";
            }

            csv << step << ','
                << ar.topples << ','
                << ar.affected << ','
                << ar.duration << ','
                << ar.max_distance << ','
                << ar.bbox_area << ','
                << ar.minx << ','
                << ar.miny << ','
                << ar.maxx << ','
                << ar.maxy << '\n';

            ++avalanche_count;
        } else {
            ++no_avalanche_steps;
        }

        // progress printout
        if ((step % logEvery) == 0) {
            std::cerr << "Step " << step << "/" << total_steps
                      << " (avalanches: " << avalanche_count
                      << ", quiet: " << no_avalanche_steps << ")\n";
        }
    }

    // ---- write summary to header ----
    csv.seekp(0, std::ios::beg);
    csv << "# Steps without avalanches: " << no_avalanche_steps << "\n";
    csv.close();

    // Save final state
    {
        std::ofstream final_ofs("sandbox_final.txt");
        sandbox.saveSandbox(final_ofs);
    }

    // ---- report ----
    std::cout << "Experiment completed.\n"
              << "Total steps: " << total_steps << "\n"
              << "Avalanches:  " << avalanche_count << "\n"
              << "No avalanche: " << no_avalanche_steps << "\n"
              << "Results saved to avalanches.csv\n";

    // Report the longest recorded avalanche that exceeded the threshold (if any)
    int bestTopples = sandbox.getBestTopples();
    if (bestTopples > 0) {
        auto br = sandbox.getBestResult();
        std::cout << "Longest avalanche (topples=" << bestTopples << ") saved to: " << framesFile << "\n";
        std::cout << "  duration=" << br.duration << ", affected=" << br.affected << ", max_distance=" << br.max_distance << "\n";
    } else {
        std::cout << "No avalanche exceeded threshold (" << animationThreshold << ") - no frames saved.\n";
    }

    return 0;
}
