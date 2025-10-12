#include "exPoints.hpp"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>

#include "generalUtils.hpp"
#include "perceptron.hpp"
#include "traininUtils.hpp"

namespace {

constexpr unsigned kDefaultSeedGeneration = 0x5EED5u;
constexpr unsigned kDefaultSeedPerceptron = 0x5EED456u;
constexpr int kTrialsBenchmark = 200000;


int expectedSign(int a, int b) {
    if (a > b) return +1;
    if (a < b) return -1;
    return 0;
}

struct EvaluationStats {
    int trials;
    int correct;
    double elapsedMs;
};

template <typename Comparator>
EvaluationStats evaluateComparator(Comparator&& comp, int trials, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, (1 << 10) - 1);

    int correct = 0;
    const auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < trials; ++i) {
        const int a = dist(rng);
        const int b = dist(rng);
        const Scalar sa = Scalar::fromInt(a);
        const Scalar sb = Scalar::fromInt(b);
        if (comp(sa, sb) == expectedSign(a, b)) ++correct;
    }
    const auto t1 = std::chrono::high_resolution_clock::now();
    const double elapsedMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return {trials, correct, elapsedMs};
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

    const auto perfectResult = evaluateComparator(
        [&perfectPerceptron](const Scalar& a, const Scalar& b) { return perfectPerceptron.compare(a, b); },
        kTrialsBenchmark,
        kDefaultSeedGeneration);
    const auto randomResult = evaluateComparator(
        [&randomPerceptron](const Scalar& a, const Scalar& b) { return randomPerceptron.compare(a, b); },
        kTrialsBenchmark,
        kDefaultSeedGeneration);
    const auto benchmarkResult = evaluateComparator(
        [](const Scalar& a, const Scalar& b) { return generalUtils::analyticalComparison(a, b); },
        kTrialsBenchmark,
        kDefaultSeedGeneration);

    generalUtils::report("Perfect Model", perfectResult.trials, perfectResult.correct, perfectResult.elapsedMs);
    generalUtils::report("Random Model", randomResult.trials, randomResult.correct, randomResult.elapsedMs);
    generalUtils::report("Benchmark", benchmarkResult.trials, benchmarkResult.correct, benchmarkResult.elapsedMs);
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
    std::cout << "\n=== Exercise 6: Generate datasets and train perceptron ===\n";
    const int P1 = 500;
    const int P2 = 2000;

    std::cout << "Generating dataset with P = " << P1 << "\n";
    const Dataset ds1 = generateDataset(P1, kDefaultSeedGeneration);
    std::cout << "Generating dataset with P = " << P2 << "\n";
    const Dataset ds2 = generateDataset(P2, kDefaultSeedGeneration + 1);

    auto train_and_summarize = [](const Dataset& ds, int P, const std::string& logfile) {
        Perceptron20 perceptron(kDefaultSeedPerceptron);
        std::cout << "Training perceptron on dataset with P = " << P << '\n';
        const TrainingStats stats = TrainPerceptron(perceptron, ds, 2000, logfile, 10, FileLogMode::EveryEpoch);

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

    const std::string out1 = "training_P" + std::to_string(P1) + ".log";
    const std::string out2 = "training_P" + std::to_string(P2) + ".log";

    train_and_summarize(ds1, P1, out1);
    train_and_summarize(ds2, P2, out2);
}

void exPointSeven() {
    std::cout << "\nExercise 7 is not implemented yet.\n";
}

void exPointEight() {
    std::cout << "\nExercise 8 is not implemented yet.\n";
}

void exPointNine() {
    std::cout << "\nExercise 9 is not implemented yet.\n";
}

void comparison() {
    std::cout << "\nComparison routine is not implemented yet.\n";
}
