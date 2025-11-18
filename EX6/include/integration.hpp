#ifndef INTEGRATION_HPP
#define INTEGRATION_HPP

#include <cmath>
#include <functional>
#include <cassert>

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

} // namespace Integration

#endif // INTEGRATION_HPP
