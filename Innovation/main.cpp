// ═══════════════════════════════════════════════════════════════════════════
//  main.cpp  –  Command dispatcher
//
//  Usage:
//      bin/main train      – run the sliding-window Word2Vec pipeline
//      bin/main innovate   – run the innovation detection pipeline
// ═══════════════════════════════════════════════════════════════════════════

#include <iostream>
#include <string>
#include <cstring>

#include "runPipeline.hpp"
#include "innovation.hpp"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: main <command>\n"
                  << "  train      Run the sliding-window Word2Vec training\n"
                  << "  innovate   Run the innovation detection pipeline\n";
        return 1;
    }

    std::string cmd = argv[1];

    if (cmd == "train")    return runTrainingPipeline();
    if (cmd == "innovate") return runInnovationPipeline();

    std::cerr << "Unknown command: " << cmd << "\n";
    return 1;
}
