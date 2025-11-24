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
    std::cout << "Define a dataset and a perceptron, train using the \n";
    std::cout << "pseudoinverse rule and print the weights" << std::endl;
    
    const int bits = 10;
    const int N = 2 * bits;
    const int P = 1000;  // Training set size

    // Generate seed with clock time
    unsigned seed = static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count());
    Dataset dataset = generateDataset(P, seed, bits);

    // Create perceptron
    Perceptron<N, InstantLearningRules::PseudoInverseRule> perceptron(seed);
    
    // Train using pseudoinverse rule
    InstantLearningRules::PseudoInverseRule rule;
    perceptron.applyInstantRule(rule, dataset);  // Sets weights in one shot

    // Print weights
    const auto finalWeights = perceptron.weights();
    std::cout << "Trained weights:\n";
    for (int i = 0; i < N; ++i) {
        std::cout << "w[" << i << "] = " << finalWeights[i] << '\n';
    }
    // Print out training error
    int trainingErrors = 0;
    for (int i = 0; i < dataset.getSize(); ++i) {
        const auto& el = dataset[i];
        Vector<int> S = Vector<int>::concat(el.top.toVector(), el.bottom.toVector());
        const int out = perceptron.eval(S);
        if (out != el.label) ++trainingErrors;
    }
    std::cout << "Training errors: " << trainingErrors << " out of " << dataset.getSize() << '\n';  
}

void exPointTwo() {
    std::cout << "Executing Exercise Point Two\n";
    std::cout << "Evaluation of the pseudoinverse procedure\n\n";
    const int bits = 10;
    const int N = 2 * bits;
    const int numTrials = 1000;  
    const int testSetSize = 1000;
    std::vector<double> alphaValues;

    // Open an outfile to log results
    std::ofstream ofs("data/exPointTwo_results.csv");
    if (!ofs) {
        std::cerr << "Failed to open output file for exPointTwo results\n";
        return;
    }

    ofs << "alpha,train_error_rate,test_error_rate\n";
    
    // alpha from 0 to 1 in steps of 0.1 (11 points)
    for (double a = 0.05; a <= 1.0; a += 0.05) {
        alphaValues.push_back(a);
    }

    unsigned testSeed = static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count());
    Dataset testDataset = generateDataset(testSetSize, testSeed, bits); 

    for (int i = 0; i < numTrials; ++i) {
        for (auto alpha : alphaValues) {
            int P = static_cast<int>(alpha * N);
            // Generate fresh dataset
            unsigned seed = static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count());
            Dataset dataset = generateDataset(P, seed, bits);
            // Create perceptron and train
            Perceptron<N, InstantLearningRules::PseudoInverseRule> perceptron(seed);
            InstantLearningRules::PseudoInverseRule rule;
            perceptron.applyInstantRule(rule, dataset);
            // Evaluate training error and test error
            int trainingErrors = 0;
            for (int j = 0; j < dataset.getSize(); ++j) {
                const auto& el = dataset[j];
                Vector<int> S = Vector<int>::concat(el.top.toVector(), el.bottom.toVector());
                const int out = perceptron.eval(S);
                if (out != el.label) ++trainingErrors;
            }
            int testErrors = 0;
            for (int j = 0; j < testDataset.getSize(); ++j) {
                const auto& el = testDataset[j];
                Vector<int> S = Vector<int>::concat(el.top.toVector(), el.bottom.toVector());
                const int out = perceptron.eval(S);
                if (out != el.label) ++testErrors;
            }
            
            ofs << alpha << "," 
                << static_cast<double>(trainingErrors) / static_cast<double>(dataset.getSize()) << ","
                << static_cast<double>(testErrors) / static_cast<double>(testDataset.getSize()) << "\n";
        }
    }
    ofs.close();
    std::cout << "Results written to exPointTwo_results.csv\n";
}

void exPointThree() {
    std::cout << "Executing Exercise Point Three\n";
    
}

void exPointFour() {
    std::cout << "Executing Exercise Point Four\n";
    std::cout << "Define a dataset and a perceptron, train using the \n";
    std::cout << "Adaline rule and print the weights" << std::endl;
    
    const int bits = 10;
    const int N = 2 * bits;
    const int P = 1000;  // Training set size

    // Generate seed with clock time
    unsigned seed = static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count());
    Dataset dataset = generateDataset(P, seed, bits);

    // Create perceptronRule> perceptron(seed);
    Perceptron<N, InstantLearningRules::AdalineRule> perceptron(seed);
    
    // Train using pseudoinverse rule
    InstantLearningRules::AdalineRule rule;
    perceptron.applyInstantRule(rule, dataset);  // Sets weights in one shot

    // Print weights
    const auto finalWeights = perceptron.weights();
    std::cout << "Trained weights:\n";
    for (int i = 0; i < N; ++i) {
        std::cout << "w[" << i << "] = " << finalWeights[i] << '\n';
    }
    // Print out training error
    int trainingErrors = 0;
    for (int i = 0; i < dataset.getSize(); ++i) {
        const auto& el = dataset[i];
        Vector<int> S = Vector<int>::concat(el.top.toVector(), el.bottom.toVector());
        const int out = perceptron.eval(S);
        if (out != el.label) ++trainingErrors;
    }
    std::cout << "Training errors: " << trainingErrors << " out of " << dataset.getSize() << '\n'; 
}

void exPointFive() {
    std::cout << "Executing Exercise Point Five\n";
    std::cout << "Evaluation of the Adaline procedure\n\n";
    const int bits = 10;
    const int N = 2 * bits;
    const int numTrials = 1000;  
    const int testSetSize = 1000;
    std::vector<double> alphaValues;

    // Open an outfile to log results
    std::ofstream ofs("data/exPointFive_results.csv");
    if (!ofs) {
        std::cerr << "Failed to open output file for exPointFive results\n";
        return;
    }

    ofs << "alpha,train_error_rate,test_error_rate\n";
    
    // alpha from 0 to 2 in steps of 0.1 (20 points)
    for (double a = 0.05; a < 2.0; a += 0.05) {
        alphaValues.push_back(a);
    }

    // alpha from 2.1 to 10 in steps of 0.5 (17 points)
    for (double a = 2.0; a <= 10.0; a += 0.5) {
        alphaValues.push_back(a);
    }

    unsigned testSeed = static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count());
    Dataset testDataset = generateDataset(testSetSize, testSeed, bits); 

    for (int i = 0; i < numTrials; ++i) {
        for (auto alpha : alphaValues) {
            int P = static_cast<int>(alpha * N);
            // Generate fresh dataset
            unsigned seed = static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count());
            Dataset dataset = generateDataset(P, seed, bits);
            // Create perceptron and train
            Perceptron<N, InstantLearningRules::AdalineRule> perceptron(seed);
            InstantLearningRules::AdalineRule rule;
            perceptron.applyInstantRule(rule, dataset);
            // Evaluate training error and test error
            int trainingErrors = 0;
            for (int j = 0; j < dataset.getSize(); ++j) {
                const auto& el = dataset[j];
                Vector<int> S = Vector<int>::concat(el.top.toVector(), el.bottom.toVector());
                const int out = perceptron.eval(S);
                if (out != el.label) ++trainingErrors;
            }
            int testErrors = 0;
            for (int j = 0; j < testDataset.getSize(); ++j) {
                const auto& el = testDataset[j];
                Vector<int> S = Vector<int>::concat(el.top.toVector(), el.bottom.toVector());
                const int out = perceptron.eval(S);
                if (out != el.label) ++testErrors;
            }
            
            ofs << alpha << "," 
                << static_cast<double>(trainingErrors) / static_cast<double>(dataset.getSize()) << ","
                << static_cast<double>(testErrors) / static_cast<double>(testDataset.getSize()) << "\n";
        }
        std::cout << "Completed trial " << (i + 1) << " / " << numTrials << "\n";
    }
    ofs.close();
    std::cout << "Results written to exPointFive_results.csv\n";
}

void exPointSix() {
    std::cout << "Executing Exercise Point Six\n";
    std::cout << "Define a dataset and a perceptron, train using the \n";
    std::cout << "Bayes rule and print the weights" << std::endl;
    
    const int bits = 10;
    const int N = 2 * bits;
    const int P = 1000;  // Training set size

    // Generate seed with clock time
    unsigned seed = static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count());
    Dataset dataset = generateDataset(P, seed, bits);

    // Create perceptron
    Perceptron<N, InstantLearningRules::BayesRule> perceptron(seed);
    
    // Train using pseudoinverse rule
    InstantLearningRules::BayesRule rule;
    perceptron.applyInstantRule(rule, dataset);  // Sets weights in one shot

    // Print weights
    const auto finalWeights = perceptron.weights();
    std::cout << "Trained weights:\n";
    for (int i = 0; i < N; ++i) {
        std::cout << "w[" << i << "] = " << finalWeights[i] << '\n';
    }
    // Print out training error
    int trainingErrors = 0;
    for (int i = 0; i < dataset.getSize(); ++i) {
        const auto& el = dataset[i];
        Vector<int> S = Vector<int>::concat(el.top.toVector(), el.bottom.toVector());
        const int out = perceptron.eval(S);
        if (out != el.label) ++trainingErrors;
    }
    std::cout << "Training errors: " << trainingErrors << " out of " << dataset.getSize() << '\n';  
}

void extraPointOne() {
    std::cout << "Executing Exercise Extra Point One\n";
    std::cout << "Evaluation of the pseudoinverse procedure using Ridge instead of standard pseudoinverse\n\n";
    const int bits = 10;
    const int N = 2 * bits;
    const int numTrials = 1000;  
    const int testSetSize = 1000;
    std::vector<double> alphaValues;

    // Open an outfile to log results
    std::ofstream ofs("data/extraPointOne_results.csv");
    if (!ofs) {
        std::cerr << "Failed to open output file for extraPointOne results\n";
        return;
    }

    ofs << "alpha,train_error_rate,test_error_rate\n";
    
    // alpha from 0 to 1 in steps of 0.1 (11 points)
    for (double a = 0.05; a <= 1.0; a += 0.05) {
        alphaValues.push_back(a);
    }

    unsigned testSeed = static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count());
    Dataset testDataset = generateDataset(testSetSize, testSeed, bits); 

    for (int i = 0; i < numTrials; ++i) {
        for (auto alpha : alphaValues) {
            int P = static_cast<int>(alpha * N);
            // Generate fresh dataset
            unsigned seed = static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count());
            Dataset dataset = generateDataset(P, seed, bits);
            // Create perceptron and train
            Perceptron<N, InstantLearningRules::RidgeRegressionRule> perceptron(seed);
            InstantLearningRules::RidgeRegressionRule rule(0.001);
            perceptron.applyInstantRule(rule, dataset);
            // Evaluate training error and test error
            int trainingErrors = 0;
            for (int j = 0; j < dataset.getSize(); ++j) {
                const auto& el = dataset[j];
                Vector<int> S = Vector<int>::concat(el.top.toVector(), el.bottom.toVector());
                const int out = perceptron.eval(S);
                if (out != el.label) ++trainingErrors;
            }
            int testErrors = 0;
            for (int j = 0; j < testDataset.getSize(); ++j) {
                const auto& el = testDataset[j];
                Vector<int> S = Vector<int>::concat(el.top.toVector(), el.bottom.toVector());
                const int out = perceptron.eval(S);
                if (out != el.label) ++testErrors;
            }
            
            ofs << alpha << "," 
                << static_cast<double>(trainingErrors) / static_cast<double>(dataset.getSize()) << ","
                << static_cast<double>(testErrors) / static_cast<double>(testDataset.getSize()) << "\n";
        }
    }
    ofs.close();
    std::cout << "Results written to extraPointOne_results.csv\n";
}

void comparison() {
    std::cout << "Executing comparison function\n";
    
}
