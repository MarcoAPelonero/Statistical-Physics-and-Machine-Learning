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
    constexpr int N = 60;  // Perceptron size (2 * bits per scalar)
    constexpr int bits = 30;  // Bits per scalar
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
    const int bits = 30;
    const int N = 60;  // 2 * bits
    const int P_test = 1000;
    const int numTrials = 1000;
    std::vector<int> datasetSize;
    std::vector<double> alphaValues;

    // alpha = P/N, so P = alpha * N
    // Denser sampling for smaller alpha values
    // 0.5 to 5.0 in steps of 0.5 (10 points)
    for (double a = 0.5; a <= 5.0; a += 0.5) {
        alphaValues.push_back(a);
    }
    // 6 to 15 in steps of 1.0 (10 points)
    for (double a = 6.0; a <= 15.0; a += 1.0) {
        alphaValues.push_back(a);
    }
    // 20 to 50 in steps of 5.0 (7 points)
    for (double a = 20.0; a <= 50.0; a += 5.0) {
        alphaValues.push_back(a);
    }
    
    for (double alpha : alphaValues) {
        datasetSize.push_back(static_cast<int>(alpha * N));
    }

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
        // Train perceptron (single epoch)
        Perceptron<60, LearningRules::HebbianRule> perceptron(seedGen());
        for (auto P : datasetSize) {
            // Generate training data with a fresh seed
            Dataset trainingData = generateDataset(P, seedGen(), bits);

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
    const int bits = 30;
    const int N = 2 * bits;
    const int mcTrials = 1500;
    const int mcTestSamples = 2000;
    const unsigned mcSeed = 12345;
    std::vector<int> datasetSize;
    std::vector<double> alphaValues;
    
    // alpha from 0.5 to 50, denser sampling for smaller values
    // 0.5 to 5.0 in steps of 0.5 (10 points)
    for (double a = 0.5; a <= 5.0; a += 0.5) {
        alphaValues.push_back(a);
    }
    // 6 to 15 in steps of 1.0 (10 points)
    for (double a = 6.0; a <= 15.0; a += 1.0) {
        alphaValues.push_back(a);
    }
    // 20 to 50 in steps of 5.0 (7 points)
    for (double a = 20.0; a <= 50.0; a += 5.0) {
        alphaValues.push_back(a);
    }
    
    for (double alpha : alphaValues) {
        datasetSize.push_back(static_cast<int>(alpha * N));
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
    ofs << "alpha,epsilon_train,epsilon_theory,epsilon_mc_train,epsilon_mc_test\n";
    for (double alpha : alphaValues) {
        double eTrain = Integration::epsilon_train(alpha);
        double eTheory = Integration::epsilon_theory(alpha);
        auto mc = Integration::epsilon_mc(alpha, bits, mcTrials, mcTestSamples, mcSeed + static_cast<unsigned>(alpha * 1000));

        ofs << alpha << "," << eTrain << "," << eTheory << "," << mc.eps_train << "," << mc.eps_test << "\n";
    }
    ofs.close();
    std::cout << "Results written to " << outPath.string() << "\n";
}

void exPointFour() {
    std::cout << "Executing Exercise Point Four\n";
    std::cout << "Analyzing generalization gap (eps_test - eps_train) for higher alpha values\n";
    
    const int bits = 30;
    const int N = 2 * bits;
    const int mcTrials = 1500;
    const int mcTestSamples = 2000;
    const unsigned mcSeed = 12345;
    std::vector<int> datasetSize;
    std::vector<double> alphaValues;
    
    // alpha from 0.5 to 70, denser sampling for smaller values
    // 0.5 to 5.0 in steps of 0.5 (10 points)
    for (double a = 0.5; a <= 5.0; a += 0.5) {
        alphaValues.push_back(a);
    }
    // 6 to 15 in steps of 1.0 (10 points)
    for (double a = 6.0; a <= 15.0; a += 1.0) {
        alphaValues.push_back(a);
    }
    // 20 to 70 in steps of 5.0 (11 points)
    for (double a = 20.0; a <= 70.0; a += 5.0) {
        alphaValues.push_back(a);
    }
    
    for (double alpha : alphaValues) {
        datasetSize.push_back(static_cast<int>(alpha * N));
    }

    // Prepare output directory and CSV file
    std::filesystem::create_directories(kExtraOutputDir);
    std::filesystem::path outPath = std::filesystem::path(kExtraOutputDir) / "exPointFour_results.csv";
    std::ofstream ofs(outPath);
    if (!ofs) {
        std::cerr << "Failed to open output file: " << outPath.string() << "\n";
        return;
    }
    
    // CSV header: same as exPointThree
    ofs << "alpha,epsilon_train,epsilon_theory,epsilon_mc_train,epsilon_mc_test\n";
    
    for (double alpha : alphaValues) {
        double eTrain = Integration::epsilon_train(alpha);
        double eTheory = Integration::epsilon_theory(alpha);
        auto mc = Integration::epsilon_mc(alpha, bits, mcTrials, mcTestSamples, mcSeed + static_cast<unsigned>(alpha * 1000));

        ofs << alpha << "," << eTrain << "," << eTheory << "," << mc.eps_train << "," << mc.eps_test << "\n";
    }
    ofs.close();
    std::cout << "Results written to " << outPath.string() << "\n";
}

void exPointFive() {
    std::cout << "Executing Exercise Point Five\n";
    std::cout << "Training with Noisy Perceptron Rule until convergence\n\n";
    
    // Perform 0 temperature learning with noisy perceptron rule, for the same alpha values as in exPointTwo
    const int bits = 10;
    const int N = 2 * bits;
    const int numTrials = 100;  // Number of trials per alpha value
    const int P_test = 1000;  // Test set size
    const int maxEpochs = 10000;  // Maximum epochs for convergence
    std::vector<int> datasetSize;
    std::vector<double> alphaValues;

    // Same alpha values as exPointTwo for comparison
    for (double a = 0.5; a <= 50.0; a += 3.0) {
        alphaValues.push_back(a);     
    }
    alphaValues.push_back(50.0); // Ensure 50.0 is included
    
    
    for (double alpha : alphaValues) {
        datasetSize.push_back(static_cast<int>(alpha * N));
    }

    // Prepare output directory and CSV file
    std::filesystem::create_directories(kExtraOutputDir);
    std::filesystem::path outPath = std::filesystem::path(kExtraOutputDir) / "exPointFive_results.csv";
    std::ofstream ofs(outPath);
    if (!ofs) {
        std::cerr << "Failed to open output file: " << outPath.string() << "\n";
        return;
    }
    
    std::mt19937 seedGen(static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count()));

    // Generate test set once
    Dataset testData = generateDataset(P_test, seedGen(), bits);
    
    // CSV header matching exPointTwo format
    ofs << "trial,train_size,train_errors,train_error_rate,test_errors,test_error_rate,epochs_to_converge\n";
    
    std::cout << "Running " << numTrials << " trials for each alpha value...\n";
    
    for (int trial = 0; trial < numTrials; ++trial) {
        std::cout << "Trial " << (trial + 1) << " / " << numTrials << "\n";
        
        for (double alpha : alphaValues) {
            int P = static_cast<int>(alpha * N);
            
            // Generate fresh training data for this trial
            Dataset trainingData = generateDataset(P, seedGen(), bits);

            // Create and train perceptron
            Perceptron<N, LearningRules::NoisyPerceptronRule> perceptron(seedGen());
            TrainingStats stats = TrainPerceptron(perceptron, trainingData, maxEpochs);

            // Compute training errors after convergence (should be 0 or very low)
            int trainErrors = stats.lastEpochErrors;

            // Compute test (generalization) errors
            int testErrors = 0;
            for (int j = 0; j < testData.getSize(); ++j) {
                const auto& el = testData[j];
                Vector<int> S = Vector<int>::concat(el.top.toVector(), el.bottom.toVector());
                const int out = perceptron.eval(S);
                if (out != el.label) ++testErrors;
            }

            double trainErrRate = static_cast<double>(trainErrors) / static_cast<double>(P);
            double testErrRate = static_cast<double>(testErrors) / static_cast<double>(testData.getSize());

            // Write CSV line: trial (1-based), training size, training errors, training error rate, 
            // test errors, test error rate, epochs to converge
            ofs << (trial + 1) << ',' << P << ',' << trainErrors << ',' << trainErrRate
                << ',' << testErrors << ',' << testErrRate << ',' << stats.epochsRun << '\n';
        }
    }
    
    ofs.close();
    std::cout << "\nResults written to " << outPath.string() << "\n";
    std::cout << "Analysis complete!\n";
}

void exPointSix() {
    std::cout << "Executing Exercise Point Six\n";
    std::cout << "Training with Perceptron Rule until convergence\n\n";
    
    // Perform 0 temperature learning with  perceptron rule, for the same alpha values as in exPointTwo
    const int bits = 10;
    const int N = 2 * bits;
    const int numTrials = 1000;  // Number of trials per alpha value
    const int P_test = 1000;  // Test set size
    std::vector<int> datasetSize;
    std::vector<double> alphaValues;

    // Uniform 10 values but the same as RANGE as exPointTwo for comparison
    for (double a = 0.5; a <= 5.0; a += 0.5) {
        alphaValues.push_back(a);
    }
    // 6 to 15 in steps of 1.0 (10 points)
    for (double a = 6.0; a <= 15.0; a += 1.0) {
        alphaValues.push_back(a);
    }
    // 20 to 50 in steps of 5.0 (7 points)
    for (double a = 20.0; a <= 50.0; a += 5.0) {
        alphaValues.push_back(a);
    }
    
    
    
    for (double alpha : alphaValues) {
        datasetSize.push_back(static_cast<int>(alpha * N));
    }

    // Prepare output directory and CSV file
    std::filesystem::create_directories(kExtraOutputDir);
    std::filesystem::path outPath = std::filesystem::path(kExtraOutputDir) / "exPointSix_results.csv";
    std::ofstream ofs(outPath);
    if (!ofs) {
        std::cerr << "Failed to open output file: " << outPath.string() << "\n";
        return;
    }
    
    std::mt19937 seedGen(static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count()));

    // Generate test set once
    Dataset testData = generateDataset(P_test, seedGen(), bits);
    
    // CSV header matching exPointTwo format
    ofs << "trial,train_size,train_errors,train_error_rate,test_errors,test_error_rate,epochs_to_converge\n";
    
    std::cout << "Running " << numTrials << " trials for each alpha value...\n";
    
    for (int trial = 0; trial < numTrials; ++trial) {
        std::cout << "Trial " << (trial + 1) << " / " << numTrials << "\n";
        
        for (double alpha : alphaValues) {
            int P = static_cast<int>(alpha * N);
            
            // Generate fresh training data for this trial
            Dataset trainingData = generateDataset(P, seedGen(), bits);

            // Create and train perceptron
            Perceptron<N, LearningRules::PerceptronRule> perceptron(seedGen());
            perceptron.resetWeights();
            TrainingStats stats = TrainPerceptronOne(perceptron, trainingData);

            // Compute training errors after convergence (should be 0 or very low)
            int trainErrors = stats.lastEpochErrors;

            // Compute test (generalization) errors
            int testErrors = 0;
            for (int j = 0; j < testData.getSize(); ++j) {
                const auto& el = testData[j];
                Vector<int> S = Vector<int>::concat(el.top.toVector(), el.bottom.toVector());
                const int out = perceptron.eval(S);
                if (out != el.label) ++testErrors;
            }

            double trainErrRate = static_cast<double>(trainErrors) / static_cast<double>(P);
            double testErrRate = static_cast<double>(testErrors) / static_cast<double>(testData.getSize());

            // Write CSV line: trial (1-based), training size, training errors, training error rate, 
            // test errors, test error rate, epochs to converge
            ofs << (trial + 1) << ',' << P << ',' << trainErrors << ',' << trainErrRate
                << ',' << testErrors << ',' << testErrRate << ',' << stats.epochsRun << '\n';
        }
    }
    
    ofs.close();
    std::cout << "\nResults written to " << outPath.string() << "\n";
    std::cout << "Analysis complete!\n";
}

void comparison() {
    std::cout << "Executing comparison function\n";
    
}
