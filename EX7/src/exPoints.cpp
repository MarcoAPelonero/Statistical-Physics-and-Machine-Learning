#include "exPoints.hpp"
#include "perceptron.hpp"
#include "learningRules.hpp"
#include "traininUtils.hpp"
#include "integration.hpp"
#include "generalUtils.hpp"

#include <chrono>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <array>
#include <numeric>
#include <cmath>
#include <thread>
#include <atomic>
#include <algorithm>
#include <sstream>
#include <cstdlib>

namespace {

constexpr const char* kExtraOutputDir = "plots";
constexpr int kBits = 10;
constexpr int kN = 2 * kBits;
constexpr int kDefaultPatterns = 1000;
constexpr int kNumTrials = 1000;
constexpr int kTestSetSize = 1000;

template <typename Rule>
using Comparator = Perceptron<kN, Rule>;

} // anonymous namespace

void exPointOne() {
    std::cout << "Executing Exercise Point One\n";
    std::cout << "Define a dataset and a perceptron, train using the \n";
    std::cout << "pseudoinverse rule and print the weights" << std::endl;
    const unsigned seed = generalUtils::clockSeed();
    Dataset dataset = generateDataset(kDefaultPatterns, seed, kBits);

    Comparator<InstantLearningRules::PseudoInverseRule> perceptron(seed);
    auto rule = InstantLearningRules::PseudoInverseRule{};
    perceptron.applyInstantRule(rule, dataset);

    generalUtils::printWeightsAndTrainingErrors(perceptron, dataset);
}

void exPointTwo() {
    std::cout << "Executing Exercise Point Two\n";
    std::cout << "Evaluation of the pseudoinverse procedure\n\n";
    std::vector<double> alphaValues = generalUtils::makeRange(0.05, 1.0, 0.05, true);
    generalUtils::runAlphaSweep<kN, InstantLearningRules::PseudoInverseRule>(
        alphaValues,
        "data/exPointTwo_results.csv",
        kBits,
        kNumTrials,
        kTestSetSize,
        [] { return InstantLearningRules::PseudoInverseRule{}; });
    std::cout << "Results written to exPointTwo_results.csv\n";
}

void exPointThree() {
    std::cout << "Executing Exercise Point Three\n";
    
}

void exPointFour() {
    std::cout << "Executing Exercise Point Four\n";
    std::cout << "Define a dataset and a perceptron, train using the \n";
    std::cout << "Adaline rule and print the weights" << std::endl;
    const unsigned seed = generalUtils::clockSeed();
    Dataset dataset = generateDataset(kDefaultPatterns, seed, kBits);

    Comparator<InstantLearningRules::AdalineRule> perceptron(seed);
    auto rule = InstantLearningRules::AdalineRule{};
    perceptron.applyInstantRule(rule, dataset);

    generalUtils::printWeightsAndTrainingErrors(perceptron, dataset);
}

void exPointFive() {
    std::cout << "Executing Exercise Point Five\n";
    std::cout << "Evaluation of the Adaline procedure\n\n";
    std::vector<double> alphaValues = generalUtils::makeRange(0.05, 2.0, 0.05, false);
    generalUtils::appendRange(alphaValues, 2.0, 10.0, 0.5, true);

    generalUtils::runAlphaSweep<kN, InstantLearningRules::AdalineRule>(
        alphaValues,
        "data/exPointFive_results.csv",
        kBits,
        kNumTrials,
        kTestSetSize,
        [] { return InstantLearningRules::AdalineRule{}; },
        true);
    std::cout << "Results written to exPointFive_results.csv\n";
}

void exPointSix() {
    std::cout << "Executing Exercise Point Six\n";
    std::cout << "Define a dataset and a perceptron, train using the \n";
    std::cout << "Bayes rule and print the weights" << std::endl;
    const unsigned seed = generalUtils::clockSeed();
    Dataset dataset = generateDataset(kDefaultPatterns, seed, kBits);

    Comparator<InstantLearningRules::BayesRule> perceptron(seed);
    auto rule = InstantLearningRules::BayesRule{};
    perceptron.applyInstantRule(rule, dataset);

    generalUtils::printWeightsAndTrainingErrors(perceptron, dataset);
}

void exPointSeven() {
    std::cout << "Executing Exercise Point Seven\n";
    std::cout << "Evaluation of the Bayes procedure\n\n";
    std::vector<double> alphaValues = generalUtils::makeRange(0.05, 2.0, 0.05, false);
    generalUtils::appendRange(alphaValues, 2.0, 10.0, 0.5, true);

    generalUtils::runAlphaSweep<kN, InstantLearningRules::BayesRule>(
        alphaValues,
        "data/exPointSeven_results.csv",
        kBits,
        kNumTrials,
        kTestSetSize,
        [] { return InstantLearningRules::BayesRule{}; },
        true);
    std::cout << "Results written to exPointSeven_results.csv\n";
}

void extraPointOne() {
    std::cout << "Executing Exercise Extra Point One\n";
    std::cout << "Evaluation of the pseudoinverse procedure using Ridge instead of standard pseudoinverse\n\n";
    std::vector<double> alphaValues = generalUtils::makeRange(0.05, 1.0, 0.05, true);

    generalUtils::runAlphaSweep<kN, InstantLearningRules::RidgeRegressionRule>(
        alphaValues,
        "data/extraPointOne_results.csv",
        kBits,
        kNumTrials,
        kTestSetSize,
        [] { return InstantLearningRules::RidgeRegressionRule(0.001); });
    std::cout << "Results written to extraPointOne_results.csv\n";
}

void comparison() {
    std::cout << "Executing comparison function\n";
    
}
