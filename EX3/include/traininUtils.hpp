#ifndef TRAINING_UTILS_HPP
#define TRAINING_UTILS_HPP

#include "vector.hpp"
#include "perceptron.hpp"

#include <random>
#include <string>
#include <optional>


struct DatasetElement {
    Scalar top;
    Scalar bottom;
    int label; 
};

class Dataset {
public:
    Dataset() : size_(0), capacity_(0), data_(nullptr) {}
    ~Dataset() { delete[] data_; }

    void add(const Scalar& top, const Scalar& bottom, int label) {
        if (size_ == capacity_) {
            resize();
        }
        data_[size_++] = {top, bottom, label};
    }

    int getSize() const { return size_; }
    const DatasetElement& operator[](int idx) const {
        assert(0 <= idx && idx < size_);
        return data_[idx];
    }
    DatasetElement& operator[](int idx) {
        assert(0 <= idx && idx < size_);
        return data_[idx];
    }
private:
    int size_;
    int capacity_;
    DatasetElement* data_;

    void resize() {
        int newCapacity = (capacity_ == 0) ? 1 : capacity_ * 2;
        DatasetElement* newData = new DatasetElement[newCapacity];
        for (int i = 0; i < size_; ++i) {
            newData[i] = data_[i];
        }
        delete[] data_;
        data_ = newData;
        capacity_ = newCapacity;
    }
};

// Generate a dataset of P examples. Each example is a pair of Scalars (top,bottom)
// together with the correct label: +1 if top>bottom, -1 if top<bottom, 0 if equal.
// The generator uses an optional seed for reproducibility.
// requires: #include <optional> (add to top of this header)
// bits: number of bits per Scalar (default 10)
Dataset generateDataset(int P, std::optional<unsigned> seed = std::nullopt, int bits = 10);

struct TrainingStats {
    int epochsRun = 0;
    int lastEpochErrors = 0;
};

enum class FileLogMode {
    EveryEpoch,
    MatchLogEvery
};

// Train the perceptron on dataset `ds` for up to `maxEpochs` epochs.
// If `outFilename` is non-empty an output file will be created and for each
// epoch a line will be written containing: epoch w0 w1 ... w(N-1) errors
// `log_every` controls console logging frequency (0 = no console logs).
// This is a function template so it works with Perceptron<N> for any N.
// Note: `dist` is forwarded to the update routine and preserved intentionally.
#include <fstream>
#include <iostream>
#include <cmath>

template <int N>
TrainingStats TrainPerceptron(Perceptron<N>& p, const Dataset& ds, int maxEpochs,
                              const std::string& outFilename = std::string(),
                              int log_every = 1,
                              FileLogMode fileLogMode = FileLogMode::EveryEpoch,
                              std::normal_distribution<double> dist = std::normal_distribution<double>(0.0, std::sqrt(50.0))) {
    TrainingStats stats{};
    std::ofstream ofs;
    if (!outFilename.empty()) {
        ofs.open(outFilename, std::ios::out);
        if (!ofs) {
            std::cerr << "Warning: could not open output file '" << outFilename << "' for writing\n";
        }
    }

    const double scale = 1.0 / std::sqrt(static_cast<double>(N));

    if (ofs) {
        // Compute the out for the whole dataset and compute the initial error count
        int correct = 0;
        for (int i = 0; i < ds.getSize(); ++i) {
            const auto& el = ds[i];
            Vector<int> S = Vector<int>::concat(el.top.toVector(), el.bottom.toVector());
            const int out = p.eval(S);
            if (out == el.label) {
                ++correct;
            }
        }
        // The print should match the one in the epoch loop below
        int total = ds.getSize();
        int residualErrors = total - correct;
        ofs << 0 << ' ' << residualErrors;
        const auto weightsSnapshot = p.weights();
        for (int j = 0; j < N; ++j) ofs << ' ' << weightsSnapshot[j];
        ofs << '\n';
    }

    for (int epoch = 0; epoch < maxEpochs; ++epoch) {
        stats.epochsRun = epoch + 1;
        int errors = 0;
        for (int i = 0; i < ds.getSize(); ++i) {
            const auto& el = ds[i];
            Vector<int> S = Vector<int>::concat(el.top.toVector(), el.bottom.toVector());
            const int out = p.eval(S);
            if (out != el.label) {
                if (el.label != 0) {
                    p.applyUpdate(S, el.label, scale, dist);
                }
                ++errors;
            }
        }

        stats.lastEpochErrors = errors;

        if (ofs) {
            const bool matchesInterval = (log_every > 0) && (epoch % log_every == 0);
            const bool shouldLog = (fileLogMode == FileLogMode::EveryEpoch) ||
                                   (fileLogMode == FileLogMode::MatchLogEvery && matchesInterval);
            const bool finalEpoch = (errors == 0) || (epoch + 1 == maxEpochs);
            if (shouldLog || finalEpoch) {
                ofs << epoch << ' ' << errors;
                const auto weightsSnapshot = p.weights();
                for (int j = 0; j < N; ++j) ofs << ' ' << weightsSnapshot[j];
                ofs << '\n';
            }
        }

        if (log_every > 0 && (epoch % log_every == 0)) {
            std::cout << "Epoch " << epoch+1 << ": errors=" << errors << '\n';
        }

        if (errors == 0) break;
    }

    if (ofs) ofs.close();
    return stats;
}

#endif // TRAINING_UTILS_HPP