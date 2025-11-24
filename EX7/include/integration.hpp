#ifndef INTEGRATION_HPP
#define INTEGRATION_HPP

#include <cmath>
#include <functional>
#include <cassert>
#include <vector>
#include <random>
#include <cstdint>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace Integration {

// ================================================================
// Simpson rule (fixed grid, robust, deterministic)
// ================================================================
namespace detail {

inline double simpsonFixedGrid(const std::function<double(double)>& f,
                               double a, double b, int n = 2000)
{
    if (n % 2 != 0) n++;
    const double h = (b - a) / n;

    double sum = f(a) + f(b);

    for (int i = 1; i < n; i += 2)
        sum += 4.0 * f(a + i * h);

    for (int i = 2; i < n; i += 2)
        sum += 2.0 * f(a + i * h);

    return (h / 3.0) * sum;
}

} // namespace detail


// Public interface – wrapper
inline double integrate(const std::function<double(double)>& f,
                        double a, double b)
{
    return detail::simpsonFixedGrid(f, a, b, 2000);
}


// ================================================================
// Gaussian measure Dx = Normal(0,1)
// ================================================================

inline double gaussianMeasure(double x) {
    return std::exp(-0.5 * x * x) / std::sqrt(2.0 * M_PI);
}

inline double H_function(double u) {
    // H(u) = ∫_u^∞ Dy = 0.5 * erfc(u/sqrt(2))
    return 0.5 * std::erfc(u / std::sqrt(2.0));
}


// ================================================================
// Theoretical prediction epsilon(α)
// ================================================================

inline double epsilon_theory(double alpha) {
    assert(alpha > 0.0);
    double arg = std::sqrt((2.0 * alpha) / (2.0 * alpha + M_PI));
    return std::acos(arg) / M_PI;
}


// ================================================================
// Training error from the formula
// ε_train(α) = 2 ∫_0^∞ Dx H(1/√α + √(2α/π) x)
// ================================================================

inline double epsilon_train(double alpha)
{
    assert(alpha > 0.0);

    const double a = 1.0 / std::sqrt(alpha);
    const double b = std::sqrt(2.0 * alpha / M_PI);

    auto integrand = [&](double x) {
        return gaussianMeasure(x) * H_function(a + b * x);
    };

    // Gaussian tails vanish after ~6σ
    return 2.0 * integrate(integrand, 0.0, 8.0);
}

// ================================================================
// Monte Carlo "theory" for the binary-input comparator task
// ================================================================
struct MonteCarloResult {
    double eps_train;
    double eps_test;
};

inline MonteCarloResult epsilon_mc(double alpha,
                                   int bits = 30,
                                   int trials = 1000,
                                   int testSamples = 2000,
                                   unsigned seed = 12345)
{
    assert(alpha > 0.0);
    assert(bits > 0);
    assert(trials > 0);
    assert(testSamples > 0);

    const int N = 2 * bits;
    const int P = std::max(1, static_cast<int>(std::round(alpha * N)));
    const double scale = 1.0 / std::sqrt(static_cast<double>(N));

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> bitDist(0, 1);

    // Comparator teacher weights match Perceptron::perfectTeacher
    std::vector<double> w_star(N);
    for (int i = 0; i < bits; ++i) {
        const double val = static_cast<double>(1u << (bits - 1 - i));
        w_star[i] = val;
        w_star[bits + i] = -val;
    }

    double trainErrSum = 0.0;
    double testErrSum = 0.0;

    std::vector<int8_t> patternBuffer; // reused to avoid reallocations
    patternBuffer.reserve(static_cast<size_t>(P) * static_cast<size_t>(N));
    std::vector<int8_t> labels;
    labels.reserve(P);

    for (int trial = 0; trial < trials; ++trial) {
        patternBuffer.clear();
        labels.clear();

        // Hebbian-trained weights (start from zero each trial)
        std::vector<double> w(N, 0.0);

        // Generate training set and accumulate Hebbian update
        for (int p = 0; p < P; ++p) {
            double dotTeacher = 0.0;
            for (int j = 0; j < N; ++j) {
                const int s = bitDist(rng) ? 1 : -1;
                patternBuffer.push_back(static_cast<int8_t>(s));
                dotTeacher += w_star[j] * static_cast<double>(s);
            }
            int lbl = (dotTeacher > 0.0) ? +1 : -1; // tie is negligible; use +1 if ever 0
            labels.push_back(static_cast<int8_t>(lbl));

            const double coeff = scale * static_cast<double>(lbl);
            const int offset = p * N;
            for (int j = 0; j < N; ++j) {
                w[j] += coeff * static_cast<double>(patternBuffer[offset + j]);
            }
        }

        // Training error on the same set
        int trainErrors = 0;
        for (int p = 0; p < P; ++p) {
            double dot = 0.0;
            const int offset = p * N;
            for (int j = 0; j < N; ++j) {
                dot += w[j] * static_cast<double>(patternBuffer[offset + j]);
            }
            const int pred = (dot > 0.0) ? +1 : -1;
            if (pred != labels[p]) ++trainErrors;
        }
        trainErrSum += static_cast<double>(trainErrors) / static_cast<double>(P);

        // Generalization error on fresh random patterns
        int testErrors = 0;
        std::vector<int> testPattern(N);
        for (int t = 0; t < testSamples; ++t) {
            double dotTeacher = 0.0;
            for (int j = 0; j < N; ++j) {
                const int s = bitDist(rng) ? 1 : -1;
                testPattern[j] = s;
                dotTeacher += w_star[j] * static_cast<double>(s);
            }
            const int lbl = (dotTeacher > 0.0) ? +1 : -1;

            double dot = 0.0;
            for (int j = 0; j < N; ++j) dot += w[j] * static_cast<double>(testPattern[j]);
            const int pred = (dot > 0.0) ? +1 : -1;
            if (pred != lbl) ++testErrors;
        }
        testErrSum += static_cast<double>(testErrors) / static_cast<double>(testSamples);
    }

    MonteCarloResult res;
    res.eps_train = trainErrSum / static_cast<double>(trials);
    res.eps_test  = testErrSum / static_cast<double>(trials);
    return res;
}

} // namespace Integration

#endif // INTEGRATION_HPP
