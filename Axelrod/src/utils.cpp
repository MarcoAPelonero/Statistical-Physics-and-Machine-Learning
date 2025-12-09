#include "utils.hpp"

namespace {
    std::mt19937& global_rng() {
        static std::mt19937 rng(std::random_device{}());
        return rng;
    }
}

int rand_int(int lo, int hi) {
    std::uniform_int_distribution<int> dist(lo, hi);
    return dist(global_rng());
}

double rand_real_0_1() {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(global_rng());
}

double similarity(const std::vector<int>& features1, const std::vector<int>& features2) {
    if (features1.size() != features2.size()) {
        throw std::invalid_argument("Feature vectors must be of the same length.");
    }

    int matching_features = 0;
    for (size_t i = 0; i < features1.size(); ++i) {
        if (features1[i] == features2[i]) {
            ++matching_features;
        }
    }

    return static_cast<double>(matching_features) / features1.size();
}