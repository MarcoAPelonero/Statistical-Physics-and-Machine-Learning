#include "funcUtils.hpp"

double hiddenFunctionA(double x) {
    return 2.0 * x;
}

double hiddenFunctionB(double x) {
    double x2  = x * x;       
    double x4  = x2 * x2;
    double x5  = x4 * x;
    double x10 = x5 * x5;
    return 2.0 * x - 10.0 * x5 + 15.0 * x10;
}

std::vector<double> hiddenFunctionA(const std::vector<double>& xs) {
    std::vector<double> ys;
    ys.reserve(xs.size());
    for (double x : xs) {
        ys.push_back(2.0 * x);
    }
    return ys;
}

std::vector<double> hiddenFunctionB(const std::vector<double>& xs) {
    std::vector<double> ys;
    ys.reserve(xs.size());
    for (double x : xs) {
        double x2  = x * x;
        double x4  = x2 * x2;
        double x5  = x4 * x;
        double x10 = x5 * x5;
        ys.push_back(2.0 * x - 10.0 * x5 + 15.0 * x10);
    }
    return ys;
}

void hiddenFunctionA(const std::vector<double>& xs, std::vector<double>& ys) {
    ys.resize(xs.size());
    for (std::size_t i = 0; i < xs.size(); ++i) {
        ys[i] = 2.0 * xs[i];
    }
}

void hiddenFunctionB(const std::vector<double>& xs, std::vector<double>& ys) {
    ys.resize(xs.size());
    for (std::size_t i = 0; i < xs.size(); ++i) {
        double x   = xs[i];
        double x2  = x * x;
        double x4  = x2 * x2;
        double x5  = x4 * x;
        double x10 = x5 * x5;
        ys[i] = 2.0 * x - 10.0 * x5 + 15.0 * x10;
    }
}

double generateNoise(rng::GaussianRandom& ggen, double stddev) {
    return ggen(0.0, stddev);
}

std::vector<double> generateNoise(rng::GaussianRandom& ggen, std::size_t n, double stddev) {
    return ggen.next(n, 0.0, stddev);
}

double generateDataPointsA(double x, double noise_stddev) {
    rng::GaussianRandom ggen;
    double noise = generateNoise(ggen, noise_stddev);
    return hiddenFunctionA(x) + noise;
}

double generateDataPointsB(double x, double noise_stddev) {
    rng::GaussianRandom ggen;
    double noise = generateNoise(ggen, noise_stddev);
    return hiddenFunctionB(x) + noise;
}

std::vector<double> generateDataPointsA(const std::vector<double>& xs, std::size_t n, double noise_stddev) {
    rng::GaussianRandom ggen;
    std::vector<double> ys = hiddenFunctionA(xs);
    std::vector<double> noises = generateNoise(ggen, n, noise_stddev);
    for (std::size_t i = 0; i < ys.size() && i < noises.size(); ++i) {
        ys[i] += noises[i];
    }
    return ys;
}

std::vector<double> generateDataPointsB(const std::vector<double>& xs, std::size_t n, double noise_stddev) {
    rng::GaussianRandom ggen;
    std::vector<double> ys = hiddenFunctionB(xs);
    std::vector<double> noises = generateNoise(ggen, n, noise_stddev);
    for (std::size_t i = 0; i < ys.size() && i < noises.size(); ++i) {
        ys[i] += noises[i];
    }
    return ys;
}

std::function<double(double, const std::vector<double>&)>
make_polynomial(std::size_t order)
{
    return [order](double x, const std::vector<double>& theta) -> double {
        if (theta.size() != order + 1)
            throw std::invalid_argument("theta.size() must be order + 1");

        // Horner's method: ((((θ_α)x + θ_{α-1})x + ... )x + θ_0)
        double y = 0.0;
        for (std::size_t i = 0; i <= order; ++i) {
            std::size_t k = order - i;   // θ_α down to θ_0
            y = y * x + theta[k];
        }
        return y;
    };
}

std::vector<double>
evaluate_many(const std::function<double(double, const std::vector<double>&)>& f,
              const std::vector<double>& xs,
              const std::vector<double>& theta)
{
    std::vector<double> ys;
    ys.reserve(xs.size());
    for (double x : xs) ys.push_back(f(x, theta));
    return ys;
}