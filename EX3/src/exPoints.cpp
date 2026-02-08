#include "exPoints.hpp"

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

void ensureDirectory(const std::string& path) {
    try {
        std::filesystem::create_directories(path);
    } catch (const std::exception& e) {
        std::cerr << "Warning: could not create directory '" << path << "': " << e.what() << "\n";
    }
}

struct RunningStats {
    int count = 0;
    double mean = 0.0;
    double m2 = 0.0;

    void add(double value) {
        ++count;
        const double delta = value - mean;
        mean += delta / static_cast<double>(count);
        const double delta2 = value - mean;
        m2 += delta * delta2;
    }

    double meanValue() const { return mean; }

    double stddev() const {
        if (count <= 0) return 0.0;
        return std::sqrt(m2 / static_cast<double>(count));
    }
};

struct SweepResult {
    int bits = 0;
    std::vector<int> datasetSizes;
    std::vector<double> meanAccuracy;
    std::vector<double> stdAccuracy;
};

std::vector<int> datasetSizesForBits(int bits) {
    if (bits == 10) return {1, 10, 20, 50, 100, 150, 200};
    if (bits == 20) return {1, 2, 20, 40, 100, 200, 300};

    std::vector<int> base = {1, 2, 20, 40, 100, 200, 300};
    std::vector<int> scaled;
    scaled.reserve(base.size());
    const double scale = static_cast<double>(bits) / 20.0;
    int previous = 0;
    for (int value : base) {
        int scaledValue = static_cast<int>(std::round(value * scale));
        if (scaledValue < 1) scaledValue = 1;
        if (scaledValue <= previous) {
            scaledValue = previous + std::max(1, static_cast<int>(scale));
        }
        scaled.push_back(scaledValue);
        previous = scaledValue;
    }
    return scaled;
}

template <int Bits>
SweepResult runAccuracySweep(int numTrials, int maxEpochs,
                             const std::vector<int>& datasetSizes,
                             const std::string& label) {
    constexpr int N = Bits * 2;
    // const double noiseSigma = std::sqrt(50.0);
    const double noiseSigma = 0.0;

    std::vector<RunningStats> stats(datasetSizes.size());
    const int reportEvery = std::max(1, numTrials / 10);

    for (int trial = 0; trial < numTrials; ++trial) {
        Dataset testDataset = generateDataset(1000, std::nullopt, Bits);
        for (std::size_t idx = 0; idx < datasetSizes.size(); ++idx) {
            Dataset trainDataset = generateDataset(datasetSizes[idx], std::nullopt, Bits);
            Perceptron<N> student;
            std::normal_distribution<double> dist(0.0, noiseSigma);
            TrainPerceptron(student, trainDataset, maxEpochs, "", -1, FileLogMode::EveryEpoch, dist);

            int correct = 0;
            for (int j = 0; j < testDataset.getSize(); ++j) {
                const auto& el = testDataset[j];
                const int pred = student.compare(el.top, el.bottom);
                if (pred == el.label) ++correct;
            }
            const double accuracy = 100.0 * static_cast<double>(correct) / static_cast<double>(testDataset.getSize());
            stats[idx].add(accuracy);
        }
        if ((trial + 1) % reportEvery == 0 || (trial + 1) == numTrials) {
            std::cout << "\r    " << label << " trial " << (trial + 1) << "/" << numTrials << std::flush;
        }
    }
    if (numTrials > 0) {
        std::cout << "\r    " << label << " trial " << numTrials << "/" << numTrials
                  << " (done)" << std::string(10, ' ') << "\n";
    }

    SweepResult result;
    result.bits = Bits;
    result.datasetSizes = datasetSizes;
    result.meanAccuracy.reserve(datasetSizes.size());
    result.stdAccuracy.reserve(datasetSizes.size());
    for (const auto& st : stats) {
        result.meanAccuracy.push_back(st.meanValue());
        result.stdAccuracy.push_back(st.stddev());
    }
    return result;
}

int parsePositiveEnv(const char* name, int fallback) {
    if (!name) return fallback;
    if (const char* raw = std::getenv(name)) {
        try {
            const int value = std::stoi(raw);
            if (value > 0) return value;
            std::cerr << "Warning: ignoring non-positive value '" << raw
                      << "' for " << name << ", using " << fallback << ".\n";
        } catch (const std::exception& e) {
            std::cerr << "Warning: could not parse " << name << "='" << raw
                      << "': " << e.what() << ". Using " << fallback << ".\n";
        }
    }
    return fallback;
}

std::vector<double> parseSigmaListFromEnv(const char* name,
                                          const std::vector<double>& defaults) {
    if (!name) return defaults;
    const char* raw = std::getenv(name);
    if (!raw) return defaults;

    std::vector<double> values;
    std::stringstream ss(raw);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (token.empty()) continue;
        try {
            values.push_back(std::stod(token));
        } catch (const std::exception& e) {
            std::cerr << "Warning: ignoring sigma token '" << token
                      << "' (" << e.what() << ")\n";
        }
    }
    if (values.empty()) {
        std::cerr << "Warning: no valid sigma parsed from " << name
                  << ", using defaults.\n";
        return defaults;
    }
    return values;
}

std::vector<int> parsePositiveIntListEnv(const char* name,
                                         const std::vector<int>& defaults) {
    if (!name) return defaults;
    const char* raw = std::getenv(name);
    if (!raw) return defaults;

    std::vector<int> values;
    std::stringstream ss(raw);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (token.empty()) continue;
        try {
            int value = std::stoi(token);
            if (value > 0) {
                values.push_back(value);
            } else {
                std::cerr << "Warning: ignoring non-positive dataset size '"
                          << token << "' for " << name << "\n";
            }
        } catch (const std::exception& e) {
            std::cerr << "Warning: could not parse dataset size token '"
                      << token << "' (" << e.what() << ")\n";
        }
    }
    if (values.empty()) {
        std::cerr << "Warning: no valid dataset sizes parsed from " << name
                  << ", using defaults.\n";
        return defaults;
    }
    return values;
}

double dotProduct(const std::vector<double>& a, const std::vector<double>& b) {
    const std::size_t n = std::min(a.size(), b.size());
    double acc = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        acc += a[i] * b[i];
    }
    return acc;
}

std::vector<double> sampleRandomTeacherWeights(int bits, std::mt19937& rng) {
    const int N = bits * 2;
    std::normal_distribution<double> gaussian(0.0, 1.0);
    std::vector<double> weights(N, 0.0);
    double normSq = 0.0;
    for (double& w : weights) {
        w = gaussian(rng);
        normSq += w * w;
    }
    if (!(normSq > 0.0)) {
        const double base = std::sqrt(static_cast<double>(N));
        for (int i = 0; i < N; ++i) weights[i] = (i % 2 == 0 ? base : -base);
        return weights;
    }
    const double scale = std::sqrt(static_cast<double>(N) / normSq);
    for (double& w : weights) w *= scale;
    return weights;
}

Dataset generateDatasetFromTeacher(int patterns,
                                   int bits,
                                   const std::vector<double>& teacherWeights,
                                   std::mt19937& rng) {
    Dataset ds;
    const int N = bits * 2;
    if (static_cast<int>(teacherWeights.size()) < N) return ds;

    std::uniform_int_distribution<int> bitDist(0, 1);
    int generated = 0;
    while (generated < patterns) {
        Scalar top(bits);
        Scalar bottom(bits);
        double dot = 0.0;

        for (int i = 0; i < bits; ++i) {
            const int bit = bitDist(rng) ? +1 : -1;
            top[i] = bit;
            dot += teacherWeights[i] * static_cast<double>(bit);
        }
        for (int i = 0; i < bits; ++i) {
            const int bit = bitDist(rng) ? +1 : -1;
            bottom[i] = bit;
            dot += teacherWeights[bits + i] * static_cast<double>(bit);
        }

        if (std::abs(dot) <= 1e-12) {
            continue; // avoid ambiguous labels; retry sample
        }

        const int label = (dot > 0.0) ? +1 : -1;
        ds.add(top, bottom, label);
        ++generated;
    }

    return ds;
}

template <int Bits>
bool sampleVersionSpaceOnSphere(const Dataset& dataset,
                                std::mt19937& rng,
                                std::vector<double>& outWeights,
                                int maxEpochs,
                                int burnIn,
                                int mixSteps,
                                int maxDirAttempts) {
    constexpr int N = Bits * 2;
    if (dataset.getSize() == 0) return false;

    std::vector<std::vector<double>> constraints;
    constraints.reserve(dataset.getSize());
    for (int i = 0; i < dataset.getSize(); ++i) {
        const auto& el = dataset[i];
        if (el.label == 0) continue;
        std::vector<double> c(N);
        for (int j = 0; j < Bits; ++j) {
            c[j] = static_cast<double>(el.top[j]) * static_cast<double>(el.label);
        }
        for (int j = 0; j < Bits; ++j) {
            c[Bits + j] = static_cast<double>(el.bottom[j]) * static_cast<double>(el.label);
        }
        constraints.push_back(std::move(c));
    }
    if (constraints.empty()) return false;

    std::normal_distribution<double> zeroNoise(0.0, 0.0);
    TrainingStats stats{};
    const int maxRestarts = 8;
    Perceptron<N> perceptron;
    for (int attempt = 0; attempt < maxRestarts; ++attempt) {
        if (attempt > 0) {
            perceptron = Perceptron<N>();
        }
        stats = TrainPerceptron(perceptron, dataset, maxEpochs,
                                "", -1, FileLogMode::EveryEpoch, zeroNoise);
        if (stats.lastEpochErrors == 0) break;
    }
    if (stats.lastEpochErrors != 0) return false;

    auto rawWeights = perceptron.weights();
    std::vector<double> current(rawWeights.begin(), rawWeights.end());
    double norm = std::sqrt(dotProduct(current, current));
    if (!(norm > 0.0)) return false;
    for (double& v : current) v /= norm;

    std::vector<double> direction(current.size(), 0.0);
    std::vector<double> tangent(current.size(), 0.0);
    std::vector<double> candidate(current.size(), 0.0);
    std::normal_distribution<double> normal(0.0, 1.0);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);

    const double pi = std::acos(-1.0);
    const double halfPi = 0.5 * pi;
    const double boundaryTol = 1e-10;
    int totalSteps = burnIn + mixSteps;
    if (totalSteps < 0) totalSteps = 0;

    for (int step = 0; step < totalSteps; ++step) {
        bool moved = false;
        for (int attempt = 0; attempt < maxDirAttempts; ++attempt) {
            double dirNorm = 0.0;
            for (std::size_t i = 0; i < direction.size(); ++i) {
                const double val = normal(rng);
                direction[i] = val;
                dirNorm += val * val;
            }
            if (dirNorm < 1e-12) continue;

            const double projection = dotProduct(direction, current);
            for (std::size_t i = 0; i < direction.size(); ++i) {
                tangent[i] = direction[i] - projection * current[i];
            }
            double tanNorm = dotProduct(tangent, tangent);
            if (tanNorm < 1e-12) continue;
            tanNorm = std::sqrt(tanNorm);
            for (double& v : tangent) v /= tanNorm;

            double globalLow = -pi;
            double globalHigh = pi;
            bool valid = true;
            for (const auto& constraint : constraints) {
                const double a = dotProduct(current, constraint);
                const double b = dotProduct(tangent, constraint);
                if (a <= boundaryTol) {
                    valid = false;
                    break;
                }
                const double theta0 = std::atan2(b, a);
                const double localLow = theta0 - halfPi;
                const double localHigh = theta0 + halfPi;
                if (localLow > globalLow) globalLow = localLow;
                if (localHigh < globalHigh) globalHigh = localHigh;
                if (globalLow >= globalHigh) {
                    valid = false;
                    break;
                }
            }
            if (!valid) continue;

            double width = globalHigh - globalLow;
            double slack = std::min(1e-9, width * 1e-6);
            double low = globalLow + slack;
            double high = globalHigh - slack;
            if (low >= high) {
                low = globalLow;
                high = globalHigh;
            }
            const double theta = low + uniform(rng) * (high - low);
            const double cosTheta = std::cos(theta);
            const double sinTheta = std::sin(theta);
            for (std::size_t i = 0; i < current.size(); ++i) {
                candidate[i] = cosTheta * current[i] + sinTheta * tangent[i];
            }

            bool ok = true;
            for (const auto& constraint : constraints) {
                const double val = dotProduct(candidate, constraint);
                if (val <= boundaryTol) {
                    ok = false;
                    break;
                }
            }
            if (!ok) continue;

            current.swap(candidate);
            moved = true;
            break;
        }
        if (!moved) return false;
    }

    const double scale = std::sqrt(static_cast<double>(N));
    for (double& v : current) v *= scale;
    outWeights = current;
    return true;
}

int classifyWithWeights(const std::vector<double>& weights,
                        const Scalar& top,
                        const Scalar& bottom) {
    const int bits = top.getSize();
    if (bottom.getSize() != bits) return 0;
    if (static_cast<int>(weights.size()) < 2 * bits) return 0;
    double acc = 0.0;
    for (int i = 0; i < bits; ++i) {
        acc += weights[i] * static_cast<double>(top[i]);
    }
    for (int i = 0; i < bits; ++i) {
        acc += weights[bits + i] * static_cast<double>(bottom[i]);
    }
    if (acc > 0.0) return +1;
    if (acc < 0.0) return -1;
    return 0;
}

} // namespace

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

void extraPointOne() {
    std::cout << "\n=== Extra Point 1: Noise sweep on 10-bit comparator (repeated trials) ===\n";
    constexpr int bits = 10;
    constexpr int N = bits * 2;

    const int trainingExamples = parsePositiveEnv("EXTRA_TRAIN_SIZE", 100);
    const int testExamples = parsePositiveEnv("EXTRA_TEST_SIZE", 1000);
    const int numTrials = parsePositiveEnv("EXTRA_TRIALS", 50);
    const std::vector<double> sigmas = parseSigmaListFromEnv(
        "EXTRA_SIGMAS",
        {0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, std::sqrt(50.0), 10.0});

    ensureDirectory(kExtraOutputDir);
    const std::string outputPath = std::string(kExtraOutputDir) + "/extra_point_one.csv";
    std::ofstream ofs(outputPath);
    std::ios::fmtflags csvOldFlags{};
    std::streamsize csvOldPrecision = 0;
    if (!ofs) {
        std::cerr << "Warning: unable to open '" << outputPath << "' for writing.\n";
    } else {
        csvOldFlags = ofs.flags();
        csvOldPrecision = ofs.precision();
        ofs << "sigma,train_size,test_size,trials,"
               "mean_accuracy,std_accuracy,"
               "mean_epochs,std_epochs,"
               "mean_errors,std_errors\n";
        ofs << std::fixed << std::setprecision(6);
    }

    const auto oldFlags = std::cout.flags();
    const std::streamsize oldPrecision = std::cout.precision();
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Training examples per trial: " << trainingExamples
              << ", test examples: " << testExamples
              << ", trials per sigma: " << numTrials << "\n";

    for (double sigma : sigmas) {
        RunningStats accStats;
        RunningStats epochStats;
        RunningStats errorStats;

        const int reportEvery = std::max(1, numTrials / 10);

        for (int trial = 0; trial < numTrials; ++trial) {
            Dataset trainingDataset = generateDataset(trainingExamples, std::nullopt, bits);
            Dataset testDataset = generateDataset(testExamples, std::nullopt, bits);
            Perceptron<N> student;
            std::normal_distribution<double> dist(0.0, sigma);

            TrainingStats stats = TrainPerceptron(student, trainingDataset, 20000, "", -1, FileLogMode::EveryEpoch, dist);

            int correct = 0;
            for (int i = 0; i < testDataset.getSize(); ++i) {
                const auto& el = testDataset[i];
                const int pred = student.compare(el.top, el.bottom);
                if (pred == el.label) ++correct;
            }
            const double accuracy = 100.0 * static_cast<double>(correct) / static_cast<double>(testDataset.getSize());

            accStats.add(accuracy);
            epochStats.add(static_cast<double>(stats.epochsRun));
            errorStats.add(static_cast<double>(stats.lastEpochErrors));

            if ((trial + 1) % reportEvery == 0 || (trial + 1) == numTrials) {
                std::cout << "\r  sigma=" << std::setw(7) << sigma
                          << " trial " << std::setw(4) << (trial + 1) << "/" << numTrials
                          << std::flush;
            }
        }
        std::cout << "\r  sigma=" << std::setw(7) << sigma
                  << " mean accuracy=" << std::setw(6) << accStats.meanValue() << "% ± "
                  << std::setw(5) << accStats.stddev()
                  << ", mean epochs=" << std::setw(6) << epochStats.meanValue() << " ± "
                  << std::setw(5) << epochStats.stddev()
                  << ", mean errors=" << std::setw(6) << errorStats.meanValue() << " ± "
                  << std::setw(5) << errorStats.stddev()
                  << std::string(10, ' ') << "\n";

        if (ofs) {
            ofs << sigma << "," << trainingExamples << "," << testExamples << "," << numTrials << ","
                << accStats.meanValue() << "," << accStats.stddev() << ","
                << epochStats.meanValue() << "," << epochStats.stddev() << ","
                << errorStats.meanValue() << "," << errorStats.stddev() << "\n";
        }
    }

    std::cout.flags(oldFlags);
    std::cout.precision(oldPrecision);

    if (ofs) {
        ofs.flags(csvOldFlags);
        ofs.precision(csvOldPrecision);
    }
}

void extraPointTwo() {
    std::cout << "\n=== Extra Point 2: Accuracy sweeps across bit widths ===\n";
    constexpr int numTrials = 200;
    constexpr int maxEpochs = 10000;
    const std::array<int, 1> bitWidths = {10};

    ensureDirectory(kExtraOutputDir);
    const std::string outputPath = std::string(kExtraOutputDir) + "/extra_point_two_new.csv";
    std::ofstream ofs(outputPath);
    std::ios::fmtflags csvOldFlags{};
    std::streamsize csvOldPrecision = 0;
    if (!ofs) {
        std::cerr << "Warning: unable to open '" << outputPath << "' for writing.\n";
    } else {
        csvOldFlags = ofs.flags();
        csvOldPrecision = ofs.precision();
        ofs << "bits,dataset_size,mean_accuracy,std_accuracy\n";
        ofs << std::fixed << std::setprecision(6);
    }

    const auto oldFlags = std::cout.flags();
    const std::streamsize oldPrecision = std::cout.precision();
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Running " << numTrials << " trials per configuration (maxEpochs=" << maxEpochs << ").\n";

    for (int bits : bitWidths) {
        const std::vector<int> dataSizes = datasetSizesForBits(bits);
        std::cout << "  bits=" << std::setw(3) << bits << " datasets:";
        for (int sz : dataSizes) std::cout << ' ' << sz;
        std::cout << "\n";

        SweepResult result;
        switch (bits) {
            case 10:
                result = runAccuracySweep<10>(numTrials, maxEpochs, dataSizes, "bits=10");
                break;
            case 20:
                result = runAccuracySweep<20>(numTrials, maxEpochs, dataSizes, "bits=20");
                break;
            case 40:
                result = runAccuracySweep<40>(numTrials, maxEpochs, dataSizes, "bits=40");
                break;
            case 60:
                result = runAccuracySweep<60>(numTrials, maxEpochs, dataSizes, "bits=60");
                break;
            case 80:
                result = runAccuracySweep<80>(numTrials, maxEpochs, dataSizes, "bits=80");
                break;
            case 100:
                result = runAccuracySweep<100>(numTrials, maxEpochs, dataSizes, "bits=100");
                break;
            case 120:
                result = runAccuracySweep<120>(numTrials, maxEpochs, dataSizes, "bits=120");
                break;
            default:
                std::cerr << "Unsupported bit width requested: " << bits << "\n";
                continue;
        }

        for (std::size_t idx = 0; idx < result.datasetSizes.size(); ++idx) {
            std::cout << "    P=" << std::setw(6) << result.datasetSizes[idx]
                      << " -> mean=" << std::setw(6) << result.meanAccuracy[idx]
                      << "%, std=" << std::setw(6) << result.stdAccuracy[idx] << "%\n";
            if (ofs) {
                ofs << result.bits << "," << result.datasetSizes[idx] << ","
                    << result.meanAccuracy[idx] << "," << result.stdAccuracy[idx] << "\n";
            }
        }
    }

    std::cout.flags(oldFlags);
    std::cout.precision(oldPrecision);

    if (ofs) {
        ofs.flags(csvOldFlags);
        ofs.precision(csvOldPrecision);
    }
}

void extraPointThree() {
    std::cout << "\n=== Extra Point 3: Version-space sampling via hit-and-run (N=20) ===\n";
    constexpr int bits = 10;  // N = 2*bits = 20

    // ---- knobs (env overrides keep your CLI-less workflow tidy) ----
    const int numTrials   = parsePositiveEnv("EXTRA3_TRIALS", 400);
    const int testSamples = parsePositiveEnv("EXTRA3_TEST_SIZE", 1000);
    const int burnIn      = parsePositiveEnv("EXTRA3_BURN_IN", 250);
    const int mixSteps    = parsePositiveEnv("EXTRA3_MIX_STEPS", 50);
    const int dirAttempts = parsePositiveEnv("EXTRA3_DIRECTION_ATTEMPTS", 256);
    const int maxEpochs   = parsePositiveEnv("EXTRA3_TRAIN_EPOCHS", 400);
    const std::vector<int> datasetSizes = parsePositiveIntListEnv(
        "EXTRA3_DATASET_SIZES",
        {5, 10, 15, 20, 30, 40, 50, 100, 200}
    );

    // ---- RNGs: separate streams for data vs sampler (seedable via env) ----
    unsigned samplerSeed = 0;
    if (const char* raw = std::getenv("EXTRA3_SAMPLER_SEED")) {
        try { long long v = std::stoll(raw); if (v > 0) samplerSeed = (unsigned)v; } catch (...) {}
    }
    std::mt19937 samplerRng(samplerSeed ? samplerSeed : std::mt19937(std::random_device{}())());

    unsigned dataSeed = 0;
    if (const char* raw = std::getenv("EXTRA3_DATA_SEED")) {
        try { long long v = std::stoll(raw); if (v > 0) dataSeed = (unsigned)v; } catch (...) {}
    }
    std::mt19937 dataRng(dataSeed ? dataSeed
                                  : (samplerSeed ? (samplerSeed ^ 0x9E3779B9u)
                                                 : std::mt19937(std::random_device{}())()));

    // ---- CSV out ----
    ensureDirectory(kExtraOutputDir);
    const std::string outPath = std::string(kExtraOutputDir) + "/extra_point_three.csv";
    std::ofstream ofs(outPath);
    if (!ofs) {
        std::cerr << "Warning: unable to open '" << outPath << "' for writing.\n";
    } else {
        ofs.setf(std::ios::fixed); ofs.precision(6);
        ofs << "bits,dataset_size,sampler_success_rate,mean_accuracy,std_accuracy,"
               "burn_in,mix_steps\n";
    }

    // ---- log banner ----
    auto oldFlags = std::cout.flags(); auto oldPrec = std::cout.precision();
    std::cout.setf(std::ios::fixed); std::cout.precision(3);
    std::cout << "Trials per size: " << numTrials
              << ", burnIn=" << burnIn
              << ", mixSteps=" << mixSteps
              << ", directionAttempts=" << dirAttempts
              << ", testSamples=" << testSamples << "\n";

    const int reportEvery = std::max(1, numTrials / 10);

    // ---- sweep P ----
    for (int P : datasetSizes) {
        if (P <= 0) continue;

        RunningStats accStats;
        int successCount = 0, samplerFailures = 0;

        for (int trial = 0; trial < numTrials; ++trial) {
            // (1) random isotropic teacher; (2) realizable train/test from that teacher
            const std::vector<double> teacherW = sampleRandomTeacherWeights(bits, dataRng);
            Dataset train = generateDatasetFromTeacher(P,        bits, teacherW, dataRng);
            Dataset test  = generateDatasetFromTeacher(testSamples, bits, teacherW, dataRng);

            if (train.getSize() != P || test.getSize() != testSamples) {
                ++samplerFailures;
            } else {
                // (3) sample a random consistent student J from the version space
                std::vector<double> J;
                if (!sampleVersionSpaceOnSphere<bits>(train, samplerRng, J,
                                                      maxEpochs, burnIn, mixSteps, dirAttempts)) {
                    ++samplerFailures;
                } else {
                    // (4) evaluate generalization on the teacher’s distribution
                    int correct = 0;
                    for (int i = 0; i < test.getSize(); ++i) {
                        const auto& el = test[i];
                        correct += (classifyWithWeights(J, el.top, el.bottom) == el.label);
                    }
                    const double acc = 100.0 * double(correct) / double(test.getSize());
                    accStats.add(acc);
                    ++successCount;
                }
            }

            if ((trial + 1) % reportEvery == 0 || (trial + 1) == numTrials) {
                std::cout << "\r  P=" << std::setw(4) << P
                          << " trial " << std::setw(4) << (trial + 1) << "/" << numTrials
                          << " (successes=" << std::setw(4) << successCount << ")"
                          << std::flush;
            }
        }

        const double successRate = (numTrials > 0)
            ? double(successCount) / double(numTrials) : 0.0;

        std::cout << "\r  P=" << std::setw(4) << P
                  << " -> sampler success=" << std::setw(6) << successRate
                  << " (" << successCount << "/" << numTrials << ")"
                  << ", failures=" << samplerFailures;
        if (successCount > 0) {
            std::cout << ", mean acc=" << std::setw(6) << accStats.meanValue()
                      << "% +/- " << std::setw(6) << accStats.stddev();
        } else {
            std::cout << ", no feasible samples";
        }
        std::cout << std::string(12, ' ') << "\n";

        if (ofs) {
            ofs << bits << "," << P << ","
                << successRate << ","
                << (successCount ? accStats.meanValue() : 0.0) << ","
                << (successCount ? accStats.stddev()    : 0.0) << ","
                << burnIn << "," << mixSteps << "\n";
        }
    }

    std::cout.flags(oldFlags); std::cout.precision(oldPrec);
    if (ofs) ofs.close();
    std::cout << "Wrote: " << outPath << "\n";
}

void extraPointFour() {
    std::cout << "\n=== Extra Point 4: Full accuracy sweep for 80-bit comparator ===\n";
    constexpr int bits = 80;

    // knobs (env overrides)
    const int numTrials = parsePositiveEnv("EXTRA4_TRIALS", 200);
    const int maxEpochs = parsePositiveEnv("EXTRA4_MAX_EPOCHS", 10000);

    ensureDirectory(kExtraOutputDir);
    const std::string outputPath = std::string(kExtraOutputDir) + "/extra_point_four.csv";
    std::ofstream ofs(outputPath);
    std::ios::fmtflags csvOldFlags{};
    std::streamsize csvOldPrecision = 0;
    if (!ofs) {
        std::cerr << "Warning: unable to open '" << outputPath << "' for writing.\n";
    } else {
        csvOldFlags = ofs.flags();
        csvOldPrecision = ofs.precision();
        ofs << "bits,dataset_size,mean_accuracy,std_accuracy\n";
        ofs << std::fixed << std::setprecision(6);
    }

    const auto oldFlags = std::cout.flags();
    const auto oldPrecision = std::cout.precision();
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Running " << numTrials << " trials for bits=" << bits << " (maxEpochs=" << maxEpochs << ").\n";

    const std::vector<int> dataSizes = datasetSizesForBits(bits);
    SweepResult result = runAccuracySweep<80>(numTrials, maxEpochs, dataSizes, "bits=80");

    for (std::size_t idx = 0; idx < result.datasetSizes.size(); ++idx) {
        std::cout << "  P=" << std::setw(6) << result.datasetSizes[idx]
                  << " -> mean=" << std::setw(6) << result.meanAccuracy[idx]
                  << "%, std=" << std::setw(6) << result.stdAccuracy[idx] << "%\n";
        if (ofs) {
            ofs << result.bits << "," << result.datasetSizes[idx] << ","
                << result.meanAccuracy[idx] << "," << result.stdAccuracy[idx] << "\n";
        }
    }

    std::cout.flags(oldFlags);
    std::cout.precision(oldPrecision);

    if (ofs) {
        ofs.flags(csvOldFlags);
        ofs.precision(csvOldPrecision);
        ofs.close();
        std::cout << "Wrote: " << outputPath << "\n";
    }
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
