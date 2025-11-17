#ifndef INTEGRATION_HPP
#define INTEGRATION_HPP

#include <cmath>
#include <functional>
#include <limits>
#include <cassert>

#define M_PI 3.14159265358979323846

namespace Integration {

// ============================================================================
// Generic numerical integration (adaptive Simpson)
// ============================================================================
namespace detail {

inline double simpson(const std::function<double(double)>& f,
                      double a, double b) {
    const double c = 0.5 * (a + b);
    return (b - a) * (f(a) + 4 * f(c) + f(b)) / 6.0;
}

inline double adaptiveSimpson(const std::function<double(double)>& f,
                              double a, double b,
                              double eps,
                              double whole) {
    const double c = 0.5 * (a + b);
    const double left  = simpson(f, a, c);
    const double right = simpson(f, c, b);
    const double diff = left + right - whole;

    if (std::fabs(diff) < 15.0 * eps) {
        return left + right + diff / 15.0;
    }
    return adaptiveSimpson(f, a, c, eps * 0.5, left)
         + adaptiveSimpson(f, c, b, eps * 0.5, right);
}

} // namespace detail

// Public interface
inline double integrate(const std::function<double(double)>& f,
                        double a, double b,
                        double eps = 1e-8) {
    double whole = detail::simpson(f, a, b);
    return detail::adaptiveSimpson(f, a, b, eps, whole);
}

// ============================================================================
// Gaussian measure helpers: Dx = Normal(0,1) measure
// ============================================================================
inline double gaussianMeasure(double x) {
    return std::exp(-0.5 * x * x) / std::sqrt(2.0 * M_PI);
}

inline double H_function(double u, double eps = 1e-8) {
    // H(u) = ∫_u^∞ Dy
    if (u < -10) return 1.0;   // practically entire mass
    if (u > 10)  return 0.0;   // negligible tail

    return integrate(gaussianMeasure, u, 10.0, eps);
}

// ============================================================================
// Specific analytic functions
// ============================================================================

// epsilon(α) = 1/pi arccos( sqrt(2α/(2α+π)) )
inline double epsilon_theory(double alpha) {
    assert(alpha > 0.0);
    double arg = std::sqrt((2 * alpha) / (2 * alpha + M_PI));
    return std::acos(arg) / M_PI;
}

// ε_train(α) = 2 ∫_0^∞ Dx H( 1/√α + sqrt(2α/π) x )
inline double epsilon_train(double alpha, double eps = 1e-8) {
    assert(alpha > 0.0);
    double a = 1.0 / std::sqrt(alpha);
    double b = std::sqrt(2.0 * alpha / M_PI);

    auto integrand = [&](double x) {
        return gaussianMeasure(x) * H_function(a + b * x, eps);
    };

    return 2.0 * integrate(integrand, 0.0, 10.0, eps);
}

} // namespace Integration

#endif // INTEGRATION_HPP
