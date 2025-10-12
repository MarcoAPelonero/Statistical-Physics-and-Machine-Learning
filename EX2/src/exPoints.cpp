#include "exPoints.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>

#include "generalUtils.hpp"
#include "perceptron.hpp"

namespace {

constexpr unsigned kDefaultSeed = 0x5EED5u;
constexpr int kTrialsExerciseTwo = 200000;


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

    std::mt19937 rng(kDefaultSeed);
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
        kTrialsExerciseTwo,
        kDefaultSeed);

    const auto benchmarkResult = evaluateComparator(
        [](const Scalar& a, const Scalar& b) { return generalUtils::analyticalComparison(a, b); },
        kTrialsExerciseTwo,
        kDefaultSeed);

    std::mt19937 rng(kDefaultSeed);
    std::uniform_int_distribution<int> dist(0, (1 << 10) - 1);
    int disagreements = 0;
    for (int i = 0; i < kTrialsExerciseTwo; ++i) {
        const int a = dist(rng);
        const int b = dist(rng);
        const Scalar sa = Scalar::fromInt(a);
        const Scalar sb = Scalar::fromInt(b);
        const int perceptronOut = perceptron.compare(sa, sb);
        const int benchmarkOut = generalUtils::analyticalComparison(sa, sb);
        if (perceptronOut != benchmarkOut) ++disagreements;
    }

    const auto report = [](const char* name, const EvaluationStats& result) {
        const double accuracy = static_cast<double>(result.correct) / result.trials;
        const std::streamsize oldPrecision = std::cout.precision();
        const auto oldFlags = std::cout.flags();

        std::cout << name << " -> trials: " << result.trials
                  << ", accuracy: " << std::fixed << std::setprecision(4) << (100.0 * accuracy) << "%"
                  << ", error: " << (100.0 * (1.0 - accuracy)) << "%"
                  << ", time: " << std::setprecision(3) << result.elapsedMs << " ms\n";

        std::cout.precision(oldPrecision);
        std::cout.flags(oldFlags);
    };

    report("Perceptron ", perceptronResult);
    report("Benchmark  ", benchmarkResult);

    const std::streamsize oldPrecision = std::cout.precision();
    const auto oldFlags = std::cout.flags();
    const double disagreementRate = static_cast<double>(disagreements) / kTrialsExerciseTwo;
    std::cout << "Disagreement rate (perceptron vs benchmark): "
              << std::fixed << std::setprecision(4) << (100.0 * disagreementRate) << "%\n";
    std::cout.precision(oldPrecision);
    std::cout.flags(oldFlags);
}

void exPointThree() {
    std::cout << "\nExercise 3 is not implemented yet.\n";
}

void exPointFour() {
    std::cout << "\nExercise 4 is not implemented yet.\n";
}

void exPointFive() {
    std::cout << "\nExercise 5 is not implemented yet.\n";
}

void exPointSix() {
    std::cout << "\nExercise 6 is not implemented yet.\n";
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
