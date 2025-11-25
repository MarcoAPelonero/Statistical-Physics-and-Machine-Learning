#ifndef GENERAL_UTILS_HPP
#define GENERAL_UTILS_HPP

#include <fstream>
#include <iostream>
#include <vector>
#include <string>

#include "perceptron.hpp"
#include "traininUtils.hpp"
#include "vector.hpp"

namespace generalUtils {

int analyticalComparison(const Scalar& top, const Scalar& bottom);

// Print a standardized evaluation report line.
// name: label to print, trials: number of trials, correct: number of correct outcomes,
// elapsedMs: total elapsed time in milliseconds.
void report(const char* name, int trials, int correct, double elapsedMs);

// Return a clock-based seed coerced to unsigned.
unsigned clockSeed();

// Append evenly spaced values in [start, end) or [start, end] depending on includeEnd.
void appendRange(std::vector<double>& dest, double start, double end, double step, bool includeEnd);

// Helper that returns a populated vector<double> with the given range.
std::vector<double> makeRange(double start, double end, double step, bool includeEnd);

template <int N, typename Rule>
int countErrors(const Perceptron<N, Rule>& perceptron, const Dataset& dataset) {
    int errors = 0;
    for (int i = 0; i < dataset.getSize(); ++i) {
        const auto& el = dataset[i];
        Vector<int> S = Vector<int>::concat(el.top.toVector(), el.bottom.toVector());
        const int out = perceptron.eval(S);
        if (out != el.label) ++errors;
    }
    return errors;
}

template <int N, typename Rule>
void printWeightsAndTrainingErrors(const Perceptron<N, Rule>& perceptron, const Dataset& dataset) {
    const auto finalWeights = perceptron.weights();
    std::cout << "Trained weights:\n";
    for (int i = 0; i < N; ++i) {
        std::cout << "w[" << i << "] = " << finalWeights[i] << '\n';
    }
    const int trainingErrors = countErrors(perceptron, dataset);
    std::cout << "Training errors: " << trainingErrors << " out of " << dataset.getSize() << '\n';
}

template <int N, typename Rule, typename RuleFactory>
void runAlphaSweep(const std::vector<double>& alphaValues,
                   const std::string& outputFile,
                   int bits,
                   int numTrials,
                   int testSetSize,
                   RuleFactory&& makeRule,
                   bool logTrials = false) {
    std::ofstream ofs(outputFile);
    if (!ofs) {
        std::cerr << "Failed to open output file '" << outputFile << "'\n";
        return;
    }

    ofs << "alpha,train_error_rate,test_error_rate\n";

    const unsigned testSeed = clockSeed();
    Dataset testDataset = generateDataset(testSetSize, testSeed, bits);

    for (int trial = 0; trial < numTrials; ++trial) {
        for (double alpha : alphaValues) {
            const int patterns = static_cast<int>(alpha * static_cast<double>(N));
            const unsigned seed = clockSeed();
            Dataset dataset = generateDataset(patterns, seed, bits);
            Perceptron<N, Rule> perceptron(seed);
            auto rule = makeRule();
            perceptron.applyInstantRule(rule, dataset);

            const int trainingErrors = countErrors(perceptron, dataset);
            const int testErrors = countErrors(perceptron, testDataset);
            ofs << alpha << ","
                << static_cast<double>(trainingErrors) / static_cast<double>(dataset.getSize()) << ","
                << static_cast<double>(testErrors) / static_cast<double>(testDataset.getSize()) << "\n";
        }

        if (logTrials) {
            std::cout << "Completed trial " << (trial + 1) << " / " << numTrials << "\n";
        }
    }
}

} // namespace generalUtils

#endif // GENERAL_UTILS_HPP
