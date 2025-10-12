#include "traininUtils.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>

Dataset generateDataset(int P, unsigned seed) {
    Dataset ds;
    ds = Dataset();

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, (1 << 10) - 1);

    for (int i = 0; i < P; ++i) {
        int a = 0;
        int b = 0;
        // Resample until the pair encodes a strictly ordered comparison.
        do {
            a = dist(rng);
            b = dist(rng);
        } while (a == b);

        const Scalar sa = Scalar::fromInt(a);
        const Scalar sb = Scalar::fromInt(b);
        int label = 0;
        if (a > b) label = +1;
        else if (a < b) label = -1;
        ds.add(sa, sb, label);
    }
    return ds;
}

TrainingStats TrainPerceptron(Perceptron20& p, const Dataset& ds, int maxEpochs,
                              const std::string& outFilename,
                              int log_every,
                              FileLogMode fileLogMode) {
    TrainingStats stats{};
    std::ofstream ofs;
    if (!outFilename.empty()) {
        ofs.open(outFilename, std::ios::out);
        if (!ofs) {
            std::cerr << "Warning: could not open output file '" << outFilename << "' for writing\n";
        }
    }

    const double scale = 1.0 / std::sqrt(20.0);

    for (int epoch = 0; epoch < maxEpochs; ++epoch) {
        stats.epochsRun = epoch + 1;
        int errors = 0;
        for (int i = 0; i < ds.getSize(); ++i) {
            const auto& el = ds[i];
            Vector S20 = Vector::concat(el.top.toVector(), el.bottom.toVector());
            const int out = p.eval(S20);
            if (out != el.label) {
                if (el.label != 0) {
                    p.applyUpdate(S20, el.label, scale);
                }
                ++errors;
            }
        }

        stats.lastEpochErrors = errors;

        // Write weights and errors to file if requested
        if (ofs) {
            const bool matchesInterval = (log_every > 0) && (epoch % log_every == 0);
            const bool shouldLog = (fileLogMode == FileLogMode::EveryEpoch) ||
                                   (fileLogMode == FileLogMode::MatchLogEvery && matchesInterval);
            const bool finalEpoch = (errors == 0) || (epoch + 1 == maxEpochs);
            if (shouldLog || finalEpoch) {
                ofs << epoch << ' ' << errors;
                const auto weightsSnapshot = p.weights();
                for (int j = 0; j < 20; ++j) ofs << ' ' << weightsSnapshot[j];
                ofs << '\n';
            }
        }

        // Console logging controlled by log_every (0 = none)
        if (log_every > 0 && (epoch % log_every == 0)) {
            std::cout << "Epoch " << epoch << ": errors=" << errors << '\n';
        }

        if (errors == 0) break;
    }

    if (ofs) ofs.close();
    return stats;
}
