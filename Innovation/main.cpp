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
#include "nullModel.hpp"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: main <command>\n"
                  << "  train      Run the sliding-window Word2Vec training\n"
                  << "  innovate   Run the innovation detection pipeline\n"
                  << "  nullmodel  Run the null model filtering pipeline\n"
                  << "  pipeline   Run all pipelines in sequence (train -> innovate -> nullmodel)\n";
        return 1;
    }

    std::string cmd = argv[1];

    if (cmd == "train")     return runTrainingPipeline();
    if (cmd == "innovate")  return runInnovationPipeline();
    if (cmd == "nullmodel") return runNullModelPipeline();

    if (cmd == "pipeline") {
        int res = runTrainingPipeline();
        if (res != 0) return res;
        res = runInnovationPipeline();
        if (res != 0) return res;
        res = runNullModelPipeline();
        if (res != 0) return res;
        // Run plot_innovations.py after all pipelines complete
        std::cout << "\nAll pipelines completed successfully. Generating plots...\n";
        int plotRes = std::system("python plot_innovations.py");
        return plotRes;
    }

    std::cerr << "Unknown command: " << cmd << "\n";
    return 1;
}
