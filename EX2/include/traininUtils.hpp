#ifndef TRAINING_UTILS_HPP
#define TRAINING_UTILS_HPP

#include "vector.hpp"
#include "perceptron.hpp"

#include <random>
#include <string>


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
Dataset generateDataset(int P, unsigned seed = 0x5EED5u);

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
// epoch a line will be written containing: epoch w0 w1 ... w19 errors
// `log_every` controls console logging frequency (0 = no console logs).
TrainingStats TrainPerceptron(Perceptron20& p, const Dataset& ds, int maxEpochs,
                              const std::string& outFilename = std::string(),
                              int log_every = 1,
                              FileLogMode fileLogMode = FileLogMode::EveryEpoch);

#endif // TRAINING_UTILS_HPP
