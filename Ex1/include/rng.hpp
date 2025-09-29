#pragma once
#ifndef RNG_HPP
#define RNG_HPP

#include <cstdint>
#include <random>
#include <vector>
#include <stdexcept>

namespace rng {

class RandomGenerator {
public:
    using engine_type = std::mt19937_64;

    RandomGenerator();                  
    explicit RandomGenerator(uint64_t seed);

    void seed(uint64_t seed_value);
    engine_type& engine();

protected:
    engine_type gen_;
};

class UniformRandom : public RandomGenerator {
public:
    using RandomGenerator::RandomGenerator;

    double next();
    std::vector<double> next(std::size_t n);

    double operator()() { return next(); }

private:
    std::uniform_real_distribution<double> dist_{0.0, 1.0};
};

class GaussianRandom : public RandomGenerator {
public:
    using RandomGenerator::RandomGenerator;

    double next(double mean = 0.0, double stddev = 1.0);
    std::vector<double> next(std::size_t n, double mean = 0.0, double stddev = 1.0);

    double operator()(double mean = 0.0, double stddev = 1.0) { return next(mean, stddev); }

private:
    std::normal_distribution<double> dist_{0.0, 1.0};
};

} 

#endif // RNG_HPP