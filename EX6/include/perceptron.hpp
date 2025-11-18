#ifndef PERCEPTRON_HPP
#define PERCEPTRON_HPP

#include <array>
#include <random>
#include <chrono>
#include <cassert>
#include <iostream>
#include <type_traits>
#include "vector.hpp"
#include "learningRules.hpp"

// Generic perceptron of size N.
template <int N, typename UpdateRule>
class Perceptron {
public:
    Perceptron() {
        static_assert(N > 0, "Perceptron size must be positive");
        auto now = std::chrono::system_clock::now().time_since_epoch().count();
        auto seed = static_cast<unsigned>(now & 0xFFFFFFFFu);
        rng_ = std::mt19937(seed);
        std::normal_distribution<double> dist(0.0, 1.0);
        for (double &w : J_) w = dist(rng_);
        if constexpr (std::is_constructible<UpdateRule, unsigned>::value) {
            updateRule_ = UpdateRule(seed);
        }
    }

    Perceptron(const unsigned seed) {
        rng_ = std::mt19937(seed);
        std::normal_distribution<double> dist(0.0, 1.0);
        for (double &w : J_) w = dist(rng_);
        if constexpr (std::is_constructible<UpdateRule, unsigned>::value) {
            updateRule_ = UpdateRule(seed);
        }
    }

    // Build a perfect teacher according to the specification:
    // For 1 <= i <= N/2: J*_i = 2^{N/2 - i}
    // For N/2+1 <= i <= N: J*_i = -J*_{i-N/2}
    // (Using 0-based indexing internally.)
    static Perceptron<N, UpdateRule> perfectTeacher() {
        static_assert(N % 2 == 0, "perfectTeacher requires even N");
        Perceptron<N, UpdateRule> p;
        const int half = N / 2;
        for (int i = 0; i < half; ++i) {
            // exponent = (N/2) - (i+1) = half - 1 - i
            const double value = static_cast<double>(1u << (half - 1 - i));
            p.J_[i] = value;
            p.J_[half + i] = -value;
        }
        return p;
    }

    // Backward-compatible factory used previously
    static Perceptron<N, UpdateRule> perfectComparatorNbits() {
        return perfectTeacher();
    }

    int eval(const Vector<int>& S) const {
        assert(S.getSize() == N);
        double acc = 0.0;
        for (int i = 0; i < N; ++i) acc += static_cast<double>(S[i]) * J_[i];
        if (acc > 0.0) return +1;
        if (acc < 0.0) return -1;
        return 0;
    }

    // Compare two Scalars (top|bottom -> N = top.getSize()+bottom.getSize())
    int compare(const Scalar& top, const Scalar& bottom) const {
        Vector<int> S = Vector<int>::concat(top.toVector(), bottom.toVector());
        return eval(S);
    }

    std::array<double, N> weights() const {
        std::array<double, N> out{};
        for (int i = 0; i < N; ++i) out[i] = J_[i];
        return out;
    }

    void resetWeights(double value = 0.0) {
        for (double &w : J_) {
            w = value;
        }
    }

    void applyUpdate(const Vector<int>& S, int label, double scale) {
        // Create a wrapper for the raw array that provides getSize() and operator[]
        struct WeightWrapper {
            double* data;
            int size;
            int getSize() const { return size; }
            double& operator[](int idx) { return data[idx]; }
            const double& operator[](int idx) const { return data[idx]; }
        };
        WeightWrapper wrapper{J_, N};
        updateRule_(wrapper, S, eval(S), label, scale);
    }

    void display() const {
        for (int i = 0; i < N; ++i) std::cout << "J[" << i << "] = " << J_[i] << "\n";
    }

private:
    double J_[N];
    std::mt19937 rng_{}; // persistent RNG for updates/noise
    UpdateRule updateRule_;
};

// No fixed-size aliases: use Perceptron<N> directly.

#endif // PERCEPTRON_HPP
