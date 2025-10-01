#include "rng.hpp"

namespace rng {

RandomGenerator::RandomGenerator()
: gen_(std::random_device{}()) {}

RandomGenerator::RandomGenerator(uint64_t seed)
: gen_(seed) {}

void RandomGenerator::seed(uint64_t seed_value) {
    gen_.seed(seed_value);
}

RandomGenerator::engine_type& RandomGenerator::engine() {
    return gen_;
}

double UniformRandom::next() {
    return dist_(gen_);
}

std::vector<double> UniformRandom::next(std::size_t n) {
    std::vector<double> out;
    out.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        out.push_back(dist_(gen_));
    }
    return out;
}

int UniformIntRandom::next(int min_inclusive, int max_exclusive) {
    if (min_inclusive > max_exclusive) {
        throw std::invalid_argument("UniformIntRandom::next: min_inclusive must be <= max_exclusive");
    }
    std::uniform_int_distribution<int> dist(min_inclusive, max_exclusive - 1);
    return dist(gen_);
}

std::vector<int> UniformIntRandom::next(std::size_t n, int min_inclusive, int max_exclusive) {
    if (min_inclusive > max_exclusive) {
        throw std::invalid_argument("UniformIntRandom::next: min_inclusive must be <= max_exclusive");
    }
    std::uniform_int_distribution<int> dist(min_inclusive, max_exclusive - 1);
    std::vector<int> out;
    out.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        out.push_back(dist(gen_));
    }
    return out;
}

double GaussianRandom::next(double mean, double stddev) {
    if (stddev < 0.0) {
        throw std::invalid_argument("GaussianRandom::next: stddev must be >= 0");
    }
    if (stddev == 0.0) {
        return mean; 
    }
    typename std::normal_distribution<double>::param_type p(mean, stddev);
    return dist_(gen_, p);
}

std::vector<double> GaussianRandom::next(std::size_t n, double mean, double stddev) {
    if (stddev < 0.0) {
        throw std::invalid_argument("GaussianRandom::next: stddev must be >= 0");
    }
    std::vector<double> out;
    out.reserve(n);
    typename std::normal_distribution<double>::param_type p(mean, stddev);

    for (std::size_t i = 0; i < n; ++i) {
        out.push_back(dist_(gen_, p));
    }
    return out;
}

} 