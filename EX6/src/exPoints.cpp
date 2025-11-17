#include "exPoints.hpp"
#include "perceptron.hpp"
#include "learningRules.hpp"
#include "traininUtils.hpp"
#include "integration.hpp"

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

} // anonymous namespace

void exPointOne() {
    std::cout << "Executing Exercise Point One\n";
    std::cout << "Single epoch training with Hebbian rule\n\n";
    
    // Define constants
    constexpr int N = 20;  // Perceptron size (2 * bits per scalar)
    constexpr int bits = 10;  // Bits per scalar
    const int P = 100;  // Number of training examples
    
    // Generate training dataset with random seed
    auto seed = static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count());
    std::cout << "Generating training dataset with " << P << " examples (seed: " << seed << ")...\n";
    Dataset trainingData = generateDataset(P, seed, bits);
    std::cout << "Dataset generated successfully.\n\n";
    
    // Create perceptron with Hebbian learning rule using random seed
    auto perceptronSeed = static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count());
    std::cout << "Initializing perceptron with Hebbian learning rule (seed: " << perceptronSeed << ")...\n";
    Perceptron<N, LearningRules::HebbianRule> perceptron(perceptronSeed);
    std::cout << "Perceptron initialized.\n\n";
    
    std::cout << "\n";

    std::cout << "Initial Errors before training: ";
    int initialErrors = 0;
    for (int i = 0; i < trainingData.getSize(); ++i) {
        const auto& el = trainingData[i];
        Vector<int> S = Vector<int>::concat(el.top.toVector(), el.bottom.toVector());
        const int out = perceptron.eval(S);
        if (out != el.label) {
            ++initialErrors;
        }
    }
    std::cout << initialErrors << "\n";
    
    // Train for a single epoch
    std::cout << "Performing single epoch update with Hebbian rule...\n";
    TrainingStats stats = TrainPerceptronOne(perceptron, trainingData);
    
    // Display training results
    std::cout << "\n=== Training Results ===\n";
    std::cout << "Epochs run: " << stats.epochsRun << "\n";
    std::cout << "Training errors after 1 epoch: " << stats.lastEpochErrors << "\n";
    
    // Display final weights
    std::cout << "\nFinal weights:\n";
    auto finalWeights = perceptron.weights();
    for (int i = 0; i < N; ++i) {
        std::cout << "  w[" << i << "] = " << std::fixed << std::setprecision(4) 
                  << finalWeights[i] << " ";
    }
    std::cout << "\n";
}

void exPointTwo() {
    std::cout << "Executing Exercise Point Two\n";
    const int bits = 10;
    const int P_test = 1000;
    const int numTrials = 1000;
    std::array<int, 20> datasetSize;

    for (int i = 0; i < 20; ++i) datasetSize[i] = (i + 1) * 50;

    // Use a single RNG to produce varied seeds
    std::mt19937 seedGen(static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count()));

    // Generate test set once
    Dataset testData = generateDataset(P_test, seedGen(), bits);

    // Prepare output directory and CSV file
    std::filesystem::create_directories(kExtraOutputDir);
    std::filesystem::path outPath = std::filesystem::path(kExtraOutputDir) / "exPointTwo_results.csv";
    std::ofstream ofs(outPath);
    if (!ofs) {
        std::cerr << "Failed to open output file: " << outPath.string() << "\n";
        return;
    }

    // CSV header
    ofs << "trial,train_size,train_errors,train_error_rate,test_errors,test_error_rate\n";

    for (int trial = 0; trial < numTrials; ++trial) {
        for (auto P : datasetSize) {
            // Generate training data with a fresh seed
            Dataset trainingData = generateDataset(P, seedGen(), bits);

            // Train perceptron (single epoch)
            Perceptron<20, LearningRules::HebbianRule> perceptron;
            perceptron.resetWeights();
            TrainingStats stats = TrainPerceptronOne(perceptron, trainingData);

            // Compute test (generalization) errors
            int testErrors = 0;
            for (int j = 0; j < testData.getSize(); ++j) {
                const auto& el = testData[j];
                Vector<int> S = Vector<int>::concat(el.top.toVector(), el.bottom.toVector());
                const int out = perceptron.eval(S);
                if (out != el.label) ++testErrors;
            }

            double trainErrRate = static_cast<double>(stats.lastEpochErrors) / static_cast<double>(P);
            double testErrRate = static_cast<double>(testErrors) / static_cast<double>(testData.getSize());

            // Write one CSV line: trial (1-based), training size, training errors, training error rate, test errors, test error rate
            ofs << (trial + 1) << ',' << P << ',' << stats.lastEpochErrors << ',' << trainErrRate
                << ',' << testErrors << ',' << testErrRate << '\n';
        }
    }

    ofs.close();
    std::cout << "Results written to " << outPath.string() << "\n";
}

void exPointThree() {
    std::cout << "Executing Exercise Point Three\n";
    const int bits = 10;
    std::array<int, 20> datasetSize;
    std::array<double, 20> alphaValues;
    for (int i = 0; i < 20; ++i) datasetSize[i] = (i + 1) * 50;
    for (int i = 0; i < 20; ++i) {
        alphaValues[i] = static_cast<double>(datasetSize[i]) / (2.0 * bits);
    }

    // Prepare output directory and CSV file
    std::filesystem::create_directories(kExtraOutputDir);
    std::filesystem::path outPath = std::filesystem::path(kExtraOutputDir) / "exPointThree_results.csv";
    std::ofstream ofs(outPath);
    // Use the functions from integration.hpp to compute epsilon_train and epsilon_theory
    if (!ofs) {
        std::cerr << "Failed to open output file: " << outPath.string() << "\n";
        return;
    }
    ofs << "alpha,epsilon_train,epsilon_theory\n";
    for (int i = 0; i < 20; ++i) {
        double alpha = alphaValues[i];

        double eTrain = Integration::epsilon_train(alpha);
        double eTheory = Integration::epsilon_theory(alpha);

        ofs << alpha << "," << eTrain << "," << eTheory << "\n";
    }
    ofs.close();
    std::cout << "Results written to " << outPath.string() << "\n";
}

void comparison() {
    std::cout << "Executing comparison function\n";
    
}