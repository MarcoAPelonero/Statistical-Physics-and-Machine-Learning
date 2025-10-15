#include "exPoints.hpp"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
#include <numeric>
#include <cmath>

#include "generalUtils.hpp"
#include "perceptron.hpp"
#include "traininUtils.hpp"

namespace {

constexpr unsigned kDefaultSeedGeneration = 0x5EED5u;
constexpr unsigned kDefaultSeedPerceptron = 0x5EED4223u;
constexpr int kTrialsBenchmark = 400000;


int expectedSign(int a, int b) {
    if (a > b) return +1;
    if (a < b) return -1;
    return 0;
}

struct EvaluationStats {
    // Aggregate totals across all rounds
    int rounds = 1;
    int trialsPerRound = 0;
    int trials = 0;         // total = rounds * trialsPerRound
    int correct = 0;        // total correct across all rounds
    double elapsedMs = 0.0; // total elapsed across all rounds (sum of rounds)

    // Convenience aggregate accuracy across all observations
    double accuracy = 0.0; // correct / trials

    // Per-round statistics
    double meanAccuracy = 0.0;
    double stddevAccuracy = 0.0;
    double meanElapsedMs = 0.0;  // mean time per round
    double stddevElapsedMs = 0.0;
};

template <typename Comparator>
EvaluationStats evaluateComparator(Comparator&& comp, int rounds, int trialsPerRound, unsigned seedBase) {
    std::vector<double> roundAcc(rounds, 0.0);
    std::vector<double> roundTimeMs(rounds, 0.0);

    int totalCorrect = 0;
    double totalElapsedMs = 0.0;

    for (int round = 0; round < rounds; ++round) {
        std::mt19937 rng(seedBase + static_cast<unsigned>(round));
        std::uniform_int_distribution<int> dist(0, (1 << 10) - 1);

        int correct = 0;
        const auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < trialsPerRound; ++i) {
            const int a = dist(rng);
            const int b = dist(rng);
            const Scalar sa = Scalar::fromInt(a);
            const Scalar sb = Scalar::fromInt(b);
            if (comp(sa, sb) == expectedSign(a, b)) ++correct;
        }
        const auto t1 = std::chrono::high_resolution_clock::now();
        const double elapsedMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

        roundTimeMs[round] = elapsedMs;
        roundAcc[round] = static_cast<double>(correct) / static_cast<double>(trialsPerRound);

        totalCorrect += correct;
        totalElapsedMs += elapsedMs;
    }

    const int totalTrials = rounds * trialsPerRound;

    const auto mean = [](const std::vector<double>& v) {
        if (v.empty()) return 0.0;
        const double s = std::accumulate(v.begin(), v.end(), 0.0);
        return s / static_cast<double>(v.size());
    };
    const auto stdev = [](const std::vector<double>& v, double m) {
        if (v.size() <= 1) return 0.0;
        double acc = 0.0;
        for (double x : v) {
            const double d = x - m;
            acc += d * d;
        }
        // sample standard deviation
        return std::sqrt(acc / static_cast<double>(v.size() - 1));
    };

    EvaluationStats stats;
    stats.rounds = rounds;
    stats.trialsPerRound = trialsPerRound;
    stats.trials = totalTrials;
    stats.correct = totalCorrect;
    stats.elapsedMs = totalElapsedMs;
    stats.accuracy = (totalTrials > 0) ? (static_cast<double>(totalCorrect) / static_cast<double>(totalTrials)) : 0.0;
    stats.meanAccuracy = mean(roundAcc);
    stats.stddevAccuracy = stdev(roundAcc, stats.meanAccuracy);
    stats.meanElapsedMs = mean(roundTimeMs);
    stats.stddevElapsedMs = stdev(roundTimeMs, stats.meanElapsedMs);
    return stats;
}

// Backward-compatible overload: default to a single round
template <typename Comparator>
EvaluationStats evaluateComparator(Comparator&& comp, int trials, unsigned seed) {
    return evaluateComparator(std::forward<Comparator>(comp), 1, trials, seed);
}

void printScalarBits(const char* label, const Scalar& value) {
    std::cout << label << " bits: ";
    value.printBits(std::cout);
    std::cout << "  = " << value() << '\n';
}

} // namespace

void exPointOne() {
    std::cout << "\n=== Exercise 1: Perfect perceptron set-up ===\n";

    const auto perceptron = Perceptron20::perfectComparator10bits();
    const auto weights = perceptron.weights();

    std::cout << "Synaptic weights (top inputs indices 0-9 | bottom 10-19):\n  ";
    for (int i = 0; i < 20; ++i) {
        std::cout << std::setw(6) << weights[i];
        if (i == 9) std::cout << "  |";
    }
    std::cout << "\n\n";

    std::mt19937 rng(kDefaultSeedGeneration);
    std::uniform_int_distribution<int> dist(0, (1 << 10) - 1);
    const int topInt = dist(rng);
    const int bottomInt = dist(rng);
    const Scalar top = Scalar::fromInt(topInt);
    const Scalar bottom = Scalar::fromInt(bottomInt);

    printScalarBits("Top   ", top);
    printScalarBits("Bottom", bottom);
    std::cout << '\n';

    const int sigma = perceptron.compare(top, bottom);
    const int expected = expectedSign(topInt, bottomInt);

    std::cout << "Perceptron output sigma = " << sigma
              << "  (expected sign(top - bottom) = " << expected << ")\n";
}

void exPointTwo() {
    std::cout << "\n=== Exercise 2: Test error evaluation ===\n";
    const Perceptron20 perceptron = Perceptron20::perfectComparator10bits();

    const auto perceptronResult = evaluateComparator(
        [&perceptron](const Scalar& a, const Scalar& b) { return perceptron.compare(a, b); },
        kTrialsBenchmark,
        kDefaultSeedGeneration);

    const auto benchmarkResult = evaluateComparator(
        [](const Scalar& a, const Scalar& b) { return generalUtils::analyticalComparison(a, b); },
        kTrialsBenchmark,
        kDefaultSeedGeneration);

    std::mt19937 rng(kDefaultSeedGeneration);
    std::uniform_int_distribution<int> dist(0, (1 << 10) - 1);
    int disagreements = 0;
    for (int i = 0; i < kTrialsBenchmark; ++i) {
        const int a = dist(rng);
        const int b = dist(rng);
        const Scalar sa = Scalar::fromInt(a);
        const Scalar sb = Scalar::fromInt(b);
        const int perceptronOut = perceptron.compare(sa, sb);
        const int benchmarkOut = generalUtils::analyticalComparison(sa, sb);
        if (perceptronOut != benchmarkOut) ++disagreements;
    }

    generalUtils::report("Perceptron ", perceptronResult.trials, perceptronResult.correct, perceptronResult.elapsedMs);
    generalUtils::report("Benchmark  ", benchmarkResult.trials, benchmarkResult.correct, benchmarkResult.elapsedMs);

    const std::streamsize oldPrecision = std::cout.precision();
    const auto oldFlags = std::cout.flags();
    const double disagreementRate = static_cast<double>(disagreements) / kTrialsBenchmark;
    std::cout << "Disagreement rate (perceptron vs benchmark): "
              << std::fixed << std::setprecision(4) << (100.0 * disagreementRate) << "%\n";
    std::cout.precision(oldPrecision);
    std::cout.flags(oldFlags);
}

void exPointThree() {
    std::cout << "\n=== Exercise 3: Gaussian weights initialization ===\n";
    const Perceptron20 perceptron = Perceptron20::perfectComparator10bits();

    const auto weights = perceptron.weights();
    std::cout << "Initial weights:\n";
    for (const auto& weight : weights) {
        std::cout << weight << " ";
    }
    std::cout << "\n";

    Perceptron20 randomPerceptron(kDefaultSeedPerceptron);
    const auto randomWeights = randomPerceptron.weights();
    std::cout << "Random Gaussian weights:\n";
    for (const auto& weight : randomWeights) {
        std::cout << weight << " ";
    }
    std::cout << "\n";
}

void exPointFour() {
    std::cout << "\n=== Exercise 4: Evaluation of the random model vs the perfect ===\n";
    const Perceptron20 perfectPerceptron = Perceptron20::perfectComparator10bits();
    // NOTE: Using the same seed for the perceptron RNG and the data RNG
    // (kDefaultSeed) creates an unintended correlation in this program's
    // deterministic sequence. That can cause the "random" perceptron to
    // systematically (and strangely) align with the test data, producing
    // a highly biased accuracy. For demonstration, construct the random
    // perceptron with a different seed (kDefaultSeed + 1).
    const Perceptron20 randomPerceptron(kDefaultSeedPerceptron);

    // Evaluate over multiple rounds with different RNG seeds to obtain mean and stddev
    const int rounds = 20; // number of independent rounds

    const auto perfectResult = evaluateComparator(
        [&perfectPerceptron](const Scalar& a, const Scalar& b) { return perfectPerceptron.compare(a, b); },
        rounds,
        kTrialsBenchmark,
        kDefaultSeedGeneration);
    const auto randomResult = evaluateComparator(
        [&randomPerceptron](const Scalar& a, const Scalar& b) { return randomPerceptron.compare(a, b); },
        rounds,
        kTrialsBenchmark,
        kDefaultSeedGeneration);
    const auto benchmarkResult = evaluateComparator(
        [](const Scalar& a, const Scalar& b) { return generalUtils::analyticalComparison(a, b); },
        rounds,
        kTrialsBenchmark,
        kDefaultSeedGeneration);

    // Pretty-print multi-round statistics
    auto printMulti = [](const char* name, const EvaluationStats& s) {
        const std::streamsize oldPrecision = std::cout.precision();
        const auto oldFlags = std::cout.flags();
    std::cout << name
          << " -> rounds: " << s.rounds
          << ", trials/round: " << s.trialsPerRound
          << ", mean accuracy: " << std::fixed << std::setprecision(4)
          << (100.0 * s.meanAccuracy) << "%"
          << " +/- " << (100.0 * s.stddevAccuracy) << "%"
          << ", mean time/round: " << std::setprecision(3) << s.meanElapsedMs << " +/- "
          << s.stddevElapsedMs << " ms"
          << ", aggregate accuracy: " << std::setprecision(4) << (100.0 * s.accuracy) << "%"
          << ", total time: " << std::setprecision(3) << s.elapsedMs << " ms"
          << '\n';
        std::cout.precision(oldPrecision);
        std::cout.flags(oldFlags);
    };

    printMulti("Perfect Model", perfectResult);
    printMulti("Random Model", randomResult);
    printMulti("Benchmark", benchmarkResult);
}

void exPointFive() {
    std::cout << "\n=== Exercise 5: Generate datasets and check labels ===\n";
    const Perceptron20 perfectPerceptron = Perceptron20::perfectComparator10bits();

    const int P1 = 500;
    const int P2 = 2000;

    std::cout << "Generating dataset with P = " << P1 << "\n";
    const Dataset ds1 = generateDataset(P1, kDefaultSeedGeneration);
    std::cout << "Generating dataset with P = " << P2 << "\n";
    const Dataset ds2 = generateDataset(P2, kDefaultSeedGeneration + 1);

    auto evaluateDataset = [&](const Dataset& ds, int P) {
        int matches = 0;
        for (int i = 0; i < ds.getSize(); ++i) {
            const auto& el = ds[i];
            const int out = perfectPerceptron.compare(el.top, el.bottom);
            if (out == el.label) ++matches;
        }
        std::cout << "Dataset P=" << P << ": " << matches << "/" << ds.getSize() << " matches with perfect perceptron (";
        const double pct = 100.0 * static_cast<double>(matches) / static_cast<double>(ds.getSize());
        std::cout << std::fixed << std::setprecision(2) << pct << "%)\n";
        // print first 5 samples
        const int nprint = std::min(5, ds.getSize());
        for (int i = 0; i < nprint; ++i) {
            const auto& e = ds[i];
            std::cout << "Sample " << i << ": top=" << e.top() << " bottom=" << e.bottom() << " label=" << e.label
                      << " perceptron=" << perfectPerceptron.compare(e.top, e.bottom) << "\n";
        }
    };

    evaluateDataset(ds1, P1);
    evaluateDataset(ds2, P2);
}

void exPointSix() {
    std::cout << "\n=== Exercise 6: Generate datasets and train perceptron (multi-seed runs) ===\n";
    const int P1 = 500;
    const int P2 = 2000;

    // Number of independent runs (different seeds for dataset and perceptron)
    const int runs = 50;
    const int maxEpochs = 10000;

    // Helper lambdas for simple stats
    auto mean = [](const std::vector<double>& v) {
        if (v.empty()) return 0.0;
        return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
    };
    auto stdev = [](const std::vector<double>& v, double m) {
        if (v.size() <= 1) return 0.0;
        double acc = 0.0;
        for (double x : v) { const double d = x - m; acc += d * d; }
        return std::sqrt(acc / static_cast<double>(v.size() - 1));
    };

    auto train_and_summarize = [&](const Dataset& ds, int P, unsigned perceptronSeed,
                                   const std::string& logfile) -> TrainingStats {
        Perceptron20 perceptron(perceptronSeed);
        std::cout << "Training perceptron on dataset with P = " << P
                  << " (seed p=" << perceptronSeed << ")" << '\n';

        int initCorrect = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : initCorrect)
#endif
        for (int i = 0; i < ds.getSize(); ++i) {
            const auto& el = ds[i];
            if (perceptron.compare(el.top, el.bottom) == el.label) {
                ++initCorrect;
            }
        }
        {
            const int total = ds.getSize();
            const int initErrors = total - initCorrect;
            const double initAcc = (total > 0)
                                       ? (100.0 * static_cast<double>(initCorrect) / static_cast<double>(total))
                                       : 0.0;
            const std::streamsize oldPrecision = std::cout.precision();
            const auto oldFlags = std::cout.flags();
            std::cout << "  Initial (pre-training) errors: " << initErrors
                      << ", accuracy: " << std::fixed << std::setprecision(2) << initAcc
                      << "% (" << initCorrect << '/' << total << ")\n";
            std::cout.precision(oldPrecision);
            std::cout.flags(oldFlags);
        }

        const TrainingStats stats = TrainPerceptron(perceptron, ds, maxEpochs, logfile, 100, FileLogMode::EveryEpoch);

        int correct = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : correct)
#endif
        for (int i = 0; i < ds.getSize(); ++i) {
            const auto& el = ds[i];
            if (perceptron.compare(el.top, el.bottom) == el.label) {
                ++correct;
            }
        }
        const int total = ds.getSize();
        const int residualErrors = total - correct;
        const double accuracy = (total > 0)
                                    ? (100.0 * static_cast<double>(correct) / static_cast<double>(total))
                                    : 0.0;
        const std::streamsize oldPrecision = std::cout.precision();
        const auto oldFlags = std::cout.flags();
        std::cout << "  Final epoch errors: " << stats.lastEpochErrors << '\n';
        std::cout << "  Training accuracy: " << std::fixed << std::setprecision(2) << accuracy
                  << "% (" << correct << '/' << total << "), residual errors=" << residualErrors << '\n';
        std::cout.precision(oldPrecision);
        std::cout.flags(oldFlags);

        return stats;
    };

    std::vector<double> epochsConvergedP1;
    std::vector<double> epochsConvergedP2;
    int failuresP1 = 0;
    int failuresP2 = 0;

    for (int run = 0; run < runs; ++run) {
        // Different seeds for datasets and perceptrons on every run
        const unsigned dsSeed1 = kDefaultSeedGeneration + static_cast<unsigned>(run) * 1000u + 0u;
        const unsigned dsSeed2 = kDefaultSeedGeneration + static_cast<unsigned>(run) * 1000u + 1u;
        const unsigned pSeed   = kDefaultSeedPerceptron + static_cast<unsigned>(run) * 1000u + 7u;

        std::cout << "\nRun " << (run + 1) << "/" << runs << "\n";
        std::cout << "Generating dataset with P = " << P1 << ", seed = " << dsSeed1 << "\n";
        const Dataset ds1 = generateDataset(P1, dsSeed1);
        std::cout << "Generating dataset with P = " << P2 << ", seed = " << dsSeed2 << "\n";
        const Dataset ds2 = generateDataset(P2, dsSeed2);

        const std::string out1 = (run == 0) ? (std::string("training_P") + std::to_string(P1) + ".log") : std::string();
        const std::string out2 = (run == 0) ? (std::string("training_P") + std::to_string(P2) + ".log") : std::string();

        const TrainingStats s1 = train_and_summarize(ds1, P1, pSeed, out1);
        const TrainingStats s2 = train_and_summarize(ds2, P2, pSeed + 13u, out2);

        if (s1.lastEpochErrors == 0) epochsConvergedP1.push_back(static_cast<double>(s1.epochsRun));
        else ++failuresP1;
        if (s2.lastEpochErrors == 0) epochsConvergedP2.push_back(static_cast<double>(s2.epochsRun));
        else ++failuresP2;
    }

    const double meanP1 = mean(epochsConvergedP1);
    const double stdP1  = stdev(epochsConvergedP1, meanP1);
    const double meanP2 = mean(epochsConvergedP2);
    const double stdP2  = stdev(epochsConvergedP2, meanP2);
    const std::size_t n1 = epochsConvergedP1.size();
    const std::size_t n2 = epochsConvergedP2.size();
    const double semP1 = (n1 > 0) ? (stdP1 / std::sqrt(static_cast<double>(n1))) : 0.0;
    const double semP2 = (n2 > 0) ? (stdP2 / std::sqrt(static_cast<double>(n2))) : 0.0;

    const std::streamsize oldPrecision = std::cout.precision();
    const auto oldFlags = std::cout.flags();
    std::cout << "\nSummary over " << runs << " runs (only converged runs included in mean/SEM)\n";
    std::cout << "P=" << P1 << ": converged " << n1 << "/" << runs
              << ", mean epochs to 0 error = " << std::fixed << std::setprecision(2) << meanP1
              << " +/- " << semP1 << " (SEM); failures = " << failuresP1 << "\n";
    std::cout << "P=" << P2 << ": converged " << n2 << "/" << runs
              << ", mean epochs to 0 error = " << std::fixed << std::setprecision(2) << meanP2
              << " +/- " << semP2 << " (SEM); failures = " << failuresP2 << "\n";
    std::cout.precision(oldPrecision);
    std::cout.flags(oldFlags);
}

void exPointSeven() {
    std::cout << "\n=== Exercise 7: Do the same as ex 6 but for N different datasets ===\n";
    const int Ps[8] = {100, 500, 1000, 2000, 5000, 20000, 262144};
    // const int Ps[8] = {500, 2000};
    
    auto train_and_summarize = [](const Dataset& ds, int P, const std::string& logfile) {
        Perceptron20 perceptron(kDefaultSeedPerceptron);
        std::cout << "Training perceptron on dataset with P = " << P << '\n';

        int initCorrect = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : initCorrect)
#endif
        for (int i = 0; i < ds.getSize(); ++i) {
            const auto& el = ds[i];
            if (perceptron.compare(el.top, el.bottom) == el.label) {
                ++initCorrect;
            }
        }
        {
            const int total = ds.getSize();
            const int initErrors = total - initCorrect;
            const double initAcc = (total > 0)
                                       ? (100.0 * static_cast<double>(initCorrect) / static_cast<double>(total))
                                       : 0.0;
            const std::streamsize oldPrecision = std::cout.precision();
            const auto oldFlags = std::cout.flags();
            std::cout << "  Initial (pre-training) errors: " << initErrors
                      << ", accuracy: " << std::fixed << std::setprecision(2) << initAcc
                      << "% (" << initCorrect << '/' << total << ")\n";
            std::cout.precision(oldPrecision);
            std::cout.flags(oldFlags);
        }

        const TrainingStats stats = TrainPerceptron(perceptron, ds, 10000, logfile, 100, FileLogMode::EveryEpoch);

        int correct = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : correct)
#endif
        for (int i = 0; i < ds.getSize(); ++i) {
            const auto& el = ds[i];
            if (perceptron.compare(el.top, el.bottom) == el.label) {
                ++correct;
            }
        }
        const int total = ds.getSize();
        const int residualErrors = total - correct;
        const double accuracy = (total > 0)
                                    ? (100.0 * static_cast<double>(correct) / static_cast<double>(total))
                                    : 0.0;
        const std::streamsize oldPrecision = std::cout.precision();
        const auto oldFlags = std::cout.flags();
        std::cout << "  Final epoch errors: " << stats.lastEpochErrors << '\n';
        std::cout << "  Training accuracy: " << std::fixed << std::setprecision(2) << accuracy
                  << "% (" << correct << '/' << total << "), residual errors=" << residualErrors << '\n';
        std::cout.precision(oldPrecision);
        std::cout.flags(oldFlags);
    };

    for (int i = 0; i < 8; ++i) {
        const int P = Ps[i];
        std::cout << "Generating dataset with P = " << P << "\n";
        const Dataset ds = generateDataset(P, kDefaultSeedGeneration + static_cast<unsigned>(i));
        const std::string logfile = std::string("training_P") + std::to_string(P) + ".log";
        train_and_summarize(ds, P, logfile);
    }
}

void exPointEight() {
    std::cout << "\nExercise 8 is not implemented yet.\n";
}

void exPointNine() {
    std::cout << "\nExercise 9 is not implemented yet.\n";
}

void comparison() {
    std::cout << "\n=== Comparison: Perceptron vs Analytical comparator ===\n";

    const Perceptron20 perceptron = Perceptron20::perfectComparator10bits();

    const auto perceptronResult = evaluateComparator(
        [&perceptron](const Scalar& a, const Scalar& b) { return perceptron.compare(a, b); },
        kTrialsBenchmark,
        kDefaultSeedGeneration);

    const auto benchmarkResult = evaluateComparator(
        [](const Scalar& a, const Scalar& b) { return generalUtils::analyticalComparison(a, b); },
        kTrialsBenchmark,
        kDefaultSeedGeneration);

    // Compute disagreement rate between the two methods on the same RNG stream
    std::mt19937 rng(kDefaultSeedGeneration);
    std::uniform_int_distribution<int> dist(0, (1 << 10) - 1);
    int disagreements = 0;
    for (int i = 0; i < kTrialsBenchmark; ++i) {
        const int a = dist(rng);
        const int b = dist(rng);
        const Scalar sa = Scalar::fromInt(a);
        const Scalar sb = Scalar::fromInt(b);
        const int perceptronOut = perceptron.compare(sa, sb);
        const int benchmarkOut = generalUtils::analyticalComparison(sa, sb);
        if (perceptronOut != benchmarkOut) ++disagreements;
    }

    generalUtils::report("Perceptron ", perceptronResult.trials, perceptronResult.correct, perceptronResult.elapsedMs);
    generalUtils::report("Benchmark  ", benchmarkResult.trials, benchmarkResult.correct, benchmarkResult.elapsedMs);

    const std::streamsize oldPrecision = std::cout.precision();
    const auto oldFlags = std::cout.flags();
    const double disagreementRate = static_cast<double>(disagreements) / kTrialsBenchmark;
    std::cout << "Disagreement rate: "
              << std::fixed << std::setprecision(4) << (100.0 * disagreementRate) << "%\n";
    std::cout.precision(oldPrecision);
    std::cout.flags(oldFlags);
}
