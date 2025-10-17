#include "exPoints.hpp"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
#include <array>
#include <numeric>
#include <cmath>
#include <thread>
#include <atomic>

void exPointOne() {
    std::cout << "\n=== Exercise 1: Initialize the perfect Perceptron called \"teacher\" ===\n";
    constexpr int bits = 10;
    constexpr int N = bits * 2; // top|bottom
    Perceptron<N> teacher = Perceptron<N>::perfectComparatorNbits();
    teacher.display();
    Dataset dataset = generateDataset(20000, std::nullopt, bits);
    int correct = 0;
    for (int i = 0; i < dataset.getSize(); ++i) {
        const auto& el = dataset[i];
        int pred = teacher.compare(el.top,  el.bottom);
        if (pred == el.label) {
            correct++;
        }
    }
    std::cout << "Teacher accuracy on generated dataset: "
              << static_cast<double>(correct) / static_cast<double>(dataset.getSize()) * 100.0
              << "%\n";
}

void exPointTwo() {
    std::cout << "\n=== Exercise 2: Generate a student perceptron with a seed ===\n";
    constexpr int bits = 10;
    constexpr int N = bits * 2;
    Perceptron<N> student;
    student.display();
    // Test the teacher by generating a dataset and seeing if it has 100% accuracy
}

void exPointThree() {
    constexpr int bits = 10;
    constexpr int N = bits * 2;
    std::cout << "\n=== Exercise 3: For N=" << N << ", 7 sets of data ===\n";
    std::vector<int> datasets = {1, 10, 20, 50, 100, 150, 200};
    for (int i = 0; i < static_cast<int>(datasets.size()); ++i) {
        Dataset dataset = generateDataset(datasets[i], std::nullopt, bits);
        std::cout << "Dataset " << datasets[i] << " examples generated.\n";
    }
}

void exPointFour() {
    std::cout << "\n=== Exercise 4: For each training set train the perceptron using the noisy perceptron rule ===\n";
    constexpr int bits = 10;
    constexpr int N = bits * 2;
    std::vector<int> datasets = {1, 10, 20, 50, 100, 150, 200};

    // Use small noise to avoid overwhelming the learning signal
    double std = std::sqrt(50.0);
    std::mt19937 rng(static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count() & 0xFFFFFFFFu));
    std::normal_distribution<double> dist(0.0, std);

    for (int i = 0; i < static_cast<int>(datasets.size()); ++i) {
        Dataset dataset = generateDataset(datasets[i], std::nullopt, bits);
        Perceptron<N> student;
        TrainingStats stats = TrainPerceptron(student, dataset, 20000, "", -1, FileLogMode::EveryEpoch, dist);
        std::cout << "Trained on dataset with " << datasets[i] << " examples. "
                  << "Epochs run: " << stats.epochsRun << ", Last epoch errors: " << stats.lastEpochErrors << "\n";
        std::cout <<  std::endl;
    }
}

void exPointFive() {
    std::cout << "\n=== Exercise 5: Not yet implemented ===\n";
    constexpr int bits = 10;
    constexpr int N = bits * 2;
    std::vector<int> datasets = {1, 10, 20, 50, 100, 150, 200};
    int lengthDatasets = static_cast<int>(datasets.size());
    std::array<double, 7> accuracies{};
    
    Dataset testDataset = generateDataset(1000, std::nullopt, bits);

    // Use small noise to avoid overwhelming the learning signal
    double std = std::sqrt(50.0);
    std::mt19937 rng(static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count() & 0xFFFFFFFFu));
    std::normal_distribution<double> dist(0.0, std);

    for (int i = 0; i < static_cast<int>(datasets.size()); ++i) {
        Dataset dataset = generateDataset(datasets[i], std::nullopt, bits);
        Perceptron<N> student;
        TrainingStats stats = TrainPerceptron(student, dataset, 20000, "", -1, FileLogMode::EveryEpoch, dist);
        std::cout << "Trained on dataset with " << datasets[i] << " examples. "
                  << "Epochs run: " << stats.epochsRun << ", Last epoch errors: " << stats.lastEpochErrors << "\n";
        std::cout <<  std::endl;

        int correct = 0;
        for (int j = 0; j < testDataset.getSize(); ++j) {
            const auto& el = testDataset[j];
            int pred = student.compare(el.top,  el.bottom);
            if (pred == el.label) {
                correct++;
            }
        }
        accuracies[i] = static_cast<double>(correct) / static_cast<double>(testDataset.getSize()) * 100.0;
    }
    std::cout << "Accuracies on the test dataset:\n";
    for (int i = 0; i < lengthDatasets; ++i) {
        std::cout << "Dataset " << datasets[i] << ": " << accuracies[i] << "%\n";
    }
}

void exPointSix() {
    std::cout << "\n=== Exercise 6: Accuracy average estimation ===\n";
    int numTrials = 1000;
    constexpr int bits = 10;
    constexpr int N = bits * 2;
    
    std::array<int, 7> datasets = {1, 10, 20, 50, 100, 150, 200};
    std::array<double, 7> accuracyAVG{}; // mean accuracy per dataset size
    std::array<double, 7> accuracySTD{}; // std deviation per dataset size

    // Store per-trial accuracies: [numTrials][datasetIndex]
    std::array<std::array<double, 7>, 1000> trialAccuracies{};

    std::mt19937 rng(static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count() & 0xFFFFFFFFu));
    std::normal_distribution<double> dist(0.0, std::sqrt(50.0));

    // Progress bar setup (safe for both serial and OpenMP builds)
    std::atomic<int> completed{0};
    std::atomic<bool> stopProgress{false};
    const int barWidth = 40;
    std::thread progressThread([&]() {
        using namespace std::chrono_literals;
        while (!stopProgress.load(std::memory_order_relaxed)) {
            int done = completed.load(std::memory_order_relaxed);
            double pct = (numTrials > 0) ? (100.0 * done / static_cast<double>(numTrials)) : 100.0;
            int filled = static_cast<int>((barWidth * done) / std::max(1, numTrials));
            std::cout << "\r[";
            for (int i = 0; i < barWidth; ++i) std::cout << (i < filled ? '#' : '-');
            std::cout << "] " << std::setw(6) << std::fixed << std::setprecision(2) << pct << "% ("
                      << done << "/" << numTrials << ")" << std::flush;
            std::this_thread::sleep_for(100ms);
        }
        // Final update at 100%
        int done = completed.load(std::memory_order_relaxed);
        double pct = (numTrials > 0) ? (100.0 * done / static_cast<double>(numTrials)) : 100.0;
        int filled = static_cast<int>((barWidth * done) / std::max(1, numTrials));
        std::cout << "\r[";
        for (int i = 0; i < barWidth; ++i) std::cout << (i < filled ? '#' : '-');
        std::cout << "] " << std::setw(6) << std::fixed << std::setprecision(2) << pct << "% ("
                  << done << "/" << numTrials << ")" << std::flush;
        std::cout << "\n";
        std::cout.unsetf(std::ios::floatfield);
    });

    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int R = 0; R < numTrials; ++R) {
        Dataset testDataset = generateDataset(1000, std::nullopt, bits);
        for (int i = 0; i < static_cast<int>(datasets.size()); ++i) {
            Dataset dataset = generateDataset(datasets[i], std::nullopt, bits);
            Perceptron<N> student;
            TrainPerceptron(student, dataset, 20000, "", -1, FileLogMode::EveryEpoch, dist);

            int correct = 0;
            for (int j = 0; j < testDataset.getSize(); ++j) {
                const auto& el = testDataset[j];
                int pred = student.compare(el.top, el.bottom);
                if (pred == el.label) {
                    correct++;
                }
            }
            double accuracy = static_cast<double>(correct) / static_cast<double>(testDataset.getSize()) * 100.0;
            trialAccuracies[R][i] = accuracy;
        }
        // Update progress after finishing one trial
        completed.fetch_add(1, std::memory_order_relaxed);
    }

    // Stop and join the progress thread
    stopProgress.store(true, std::memory_order_relaxed);
    if (progressThread.joinable()) progressThread.join();


    // Compute mean and standard deviation per dataset size across trials
    for (int i = 0; i < static_cast<int>(datasets.size()); ++i) {
        // mean
        double sum = 0.0;
        for (int R = 0; R < numTrials; ++R) {
            sum += trialAccuracies[R][i];
        }
        double mean = sum / static_cast<double>(numTrials);
        accuracyAVG[i] = mean;

        // std deviation (unbiased sample std or population? Using population over trials here)
        double varSum = 0.0;
        for (int R = 0; R < numTrials; ++R) {
            double diff = trialAccuracies[R][i] - mean;
            varSum += diff * diff;
        }
        double variance = varSum / static_cast<double>(numTrials); // population variance over trials
        accuracySTD[i] = std::sqrt(variance);
    }

    // Print results
    std::cout << "Dataset sizes:            ";
    for (int i = 0; i < static_cast<int>(datasets.size()); ++i) {
        std::cout << std::setw(6) << datasets[i] << (i + 1 < static_cast<int>(datasets.size()) ? " " : "\n");
    }
    std::cout << "Mean accuracy (%):       ";
    for (int i = 0; i < static_cast<int>(datasets.size()); ++i) {
        std::cout << std::setw(6) << std::fixed << std::setprecision(2) << accuracyAVG[i]
                  << (i + 1 < static_cast<int>(datasets.size()) ? " " : "\n");
    }
    std::cout.unsetf(std::ios::floatfield);
    std::cout << "Std dev of accuracy (%): ";
    for (int i = 0; i < static_cast<int>(datasets.size()); ++i) {
        std::cout << std::setw(6) << std::fixed << std::setprecision(2) << accuracySTD[i]
                  << (i + 1 < static_cast<int>(datasets.size()) ? " " : "\n");
    }
    std::cout.unsetf(std::ios::floatfield);

}

void exPointSeven() {
    std::cout << "\n=== Exercise 7: Implemented in Python ===\n";
}

void exPointEight() {
    std::cout << "\n=== Exercise 8: Same as exercise 6, but for 20 bits per number ===\n";
    int numTrials = 1000;
    constexpr int bits = 20;
    constexpr int N = bits * 2;
    
    std::array<int, 7> datasets = {1, 2, 20, 40, 100, 200, 300};
    std::array<double, 7> accuracyAVG{}; // mean accuracy per dataset size
    std::array<double, 7> accuracySTD{}; // std deviation per dataset size

    // Store per-trial accuracies: [numTrials][datasetIndex]
    std::array<std::array<double, 7>, 1000> trialAccuracies{};

    std::mt19937 rng(static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count() & 0xFFFFFFFFu));
    std::normal_distribution<double> dist(0.0, std::sqrt(50.0));

    // Progress bar setup (safe for both serial and OpenMP builds)
    std::atomic<int> completed{0};
    std::atomic<bool> stopProgress{false};
    const int barWidth = 40;
    std::thread progressThread([&]() {
        using namespace std::chrono_literals;
        while (!stopProgress.load(std::memory_order_relaxed)) {
            int done = completed.load(std::memory_order_relaxed);
            double pct = (numTrials > 0) ? (100.0 * done / static_cast<double>(numTrials)) : 100.0;
            int filled = static_cast<int>((barWidth * done) / std::max(1, numTrials));
            std::cout << "\r[";
            for (int i = 0; i < barWidth; ++i) std::cout << (i < filled ? '#' : '-');
            std::cout << "] " << std::setw(6) << std::fixed << std::setprecision(2) << pct << "% ("
                      << done << "/" << numTrials << ")" << std::flush;
            std::this_thread::sleep_for(100ms);
        }
        // Final update at 100%
        int done = completed.load(std::memory_order_relaxed);
        double pct = (numTrials > 0) ? (100.0 * done / static_cast<double>(numTrials)) : 100.0;
        int filled = static_cast<int>((barWidth * done) / std::max(1, numTrials));
        std::cout << "\r[";
        for (int i = 0; i < barWidth; ++i) std::cout << (i < filled ? '#' : '-');
        std::cout << "] " << std::setw(6) << std::fixed << std::setprecision(2) << pct << "% ("
                  << done << "/" << numTrials << ")" << std::flush;
        std::cout << "\n";
        std::cout.unsetf(std::ios::floatfield);
    });

    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int R = 0; R < numTrials; ++R) {
        Dataset testDataset = generateDataset(1000, std::nullopt, bits);
        for (int i = 0; i < static_cast<int>(datasets.size()); ++i) {
            Dataset dataset = generateDataset(datasets[i], std::nullopt, bits);
            Perceptron<N> student;
            TrainPerceptron(student, dataset, 20000, "", -1, FileLogMode::EveryEpoch, dist);

            int correct = 0;
            for (int j = 0; j < testDataset.getSize(); ++j) {
                const auto& el = testDataset[j];
                int pred = student.compare(el.top, el.bottom);
                if (pred == el.label) {
                    correct++;
                }
            }
            double accuracy = static_cast<double>(correct) / static_cast<double>(testDataset.getSize()) * 100.0;
            trialAccuracies[R][i] = accuracy;
        }
        // Update progress after finishing one trial
        completed.fetch_add(1, std::memory_order_relaxed);
    }

    // Stop and join the progress thread
    stopProgress.store(true, std::memory_order_relaxed);
    if (progressThread.joinable()) progressThread.join();


    // Compute mean and standard deviation per dataset size across trials
    for (int i = 0; i < static_cast<int>(datasets.size()); ++i) {
        // mean
        double sum = 0.0;
        for (int R = 0; R < numTrials; ++R) {
            sum += trialAccuracies[R][i];
        }
        double mean = sum / static_cast<double>(numTrials);
        accuracyAVG[i] = mean;

        // std deviation (unbiased sample std or population? Using population over trials here)
        double varSum = 0.0;
        for (int R = 0; R < numTrials; ++R) {
            double diff = trialAccuracies[R][i] - mean;
            varSum += diff * diff;
        }
        double variance = varSum / static_cast<double>(numTrials); // population variance over trials
        accuracySTD[i] = std::sqrt(variance);
    }

    // Print results
    std::cout << "Dataset sizes:            ";
    for (int i = 0; i < static_cast<int>(datasets.size()); ++i) {
        std::cout << std::setw(6) << datasets[i] << (i + 1 < static_cast<int>(datasets.size()) ? " " : "\n");
    }
    std::cout << "Mean accuracy (%):       ";
    for (int i = 0; i < static_cast<int>(datasets.size()); ++i) {
        std::cout << std::setw(6) << std::fixed << std::setprecision(2) << accuracyAVG[i]
                  << (i + 1 < static_cast<int>(datasets.size()) ? " " : "\n");
    }
    std::cout.unsetf(std::ios::floatfield);
    std::cout << "Std dev of accuracy (%): ";
    for (int i = 0; i < static_cast<int>(datasets.size()); ++i) {
        std::cout << std::setw(6) << std::fixed << std::setprecision(2) << accuracySTD[i]
                  << (i + 1 < static_cast<int>(datasets.size()) ? " " : "\n");
    }
    std::cout.unsetf(std::ios::floatfield);

}

void exPointNine() {
    std::cout << "\n=== Exercise 9: Not yet implemented ===\n";
}

void exPointTen() {
    std::cout << "\n=== Comparison: Not yet implemented ===\n";
}

void comparison() {
    std::cout << "\n=== Comparison: Not yet implemented ===\n";
}