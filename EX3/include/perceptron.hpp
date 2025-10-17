#ifndef PERCEPTRON_HPP
#define PERCEPTRON_HPP

#include <array>
#include <random>
#include <chrono>
#include <cassert>
#include <iostream>
#include "vector.hpp"

// Generic perceptron of size N.
template <int N>
class Perceptron {
public:
    Perceptron() {
        static_assert(N > 0, "Perceptron size must be positive");
        auto now = std::chrono::system_clock::now().time_since_epoch().count();
        auto seed = static_cast<unsigned>(now & 0xFFFFFFFFu);
        rng_ = std::mt19937(seed);
        std::normal_distribution<double> dist(0.0, 1.0);
        for (double &w : J_) w = dist(rng_);
    }

    Perceptron(const unsigned seed) {
        rng_ = std::mt19937(seed);
        std::normal_distribution<double> dist(0.0, 1.0);
        for (double &w : J_) w = dist(rng_);
    }

    // Build a perfect teacher according to the specification:
    // For 1 <= i <= N/2: J*_i = 2^{N/2 - i}
    // For N/2+1 <= i <= N: J*_i = -J*_{i-N/2}
    // (Using 0-based indexing internally.)
    static Perceptron<N> perfectTeacher() {
        static_assert(N % 2 == 0, "perfectTeacher requires even N");
        Perceptron<N> p;
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
    static Perceptron<N> perfectComparatorNbits() {
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

    void applyUpdate(const Vector<int>& S, int label, double scale, std::normal_distribution<double>& dist) {
        if (label == 0) return;
        const double delta = scale * static_cast<double>(label);
        // Generate noise outside the SIMD loop and reuse a persistent RNG for reproducibility.
        // IMPORTANT: Use a multiplicative factor with mean 1 so E[update] matches the pure perceptron rule.
        // If dist has mean 0 (default), we use factor = 1 + noise (mean 1). If caller passes mean mu, factor = 1 + noise retains E[factor]=1+mu.
        double noise[N];
        for (int i = 0; i < N; ++i) noise[i] = dist(rng_);
        for (int i = 0; i < N; ++i) {
            const double factor = 1.0 + noise[i];
            J_[i] += delta * static_cast<double>(S[i]) * factor;
        }
    }

    void display() const {
        for (int i = 0; i < N; ++i) std::cout << "J[" << i << "] = " << J_[i] << "\n";
    }

private:
    double J_[N];
    std::mt19937 rng_{}; // persistent RNG for updates/noise
};

// No fixed-size aliases: use Perceptron<N> directly.

#endif // PERCEPTRON_HPP
