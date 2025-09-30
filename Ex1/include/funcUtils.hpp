#ifndef FUNCUTILS_HPP
#define FUNCUTILS_HPP

#include <vector>
#include <functional>
#include <stdexcept>
#include "rng.hpp"

double hiddenFunctionA(double x);
double hiddenFunctionB(double x);

std::vector<double> hiddenFunctionA(const std::vector<double>& xs);
std::vector<double> hiddenFunctionB(const std::vector<double>& xs);

void hiddenFunctionA(const std::vector<double>& xs, std::vector<double>& ys);
void hiddenFunctionB(const std::vector<double>& xs, std::vector<double>& ys);

double generateNoise(rng::GaussianRandom& ggen, double stddev = 1.0);
std::vector<double> generateNoise(rng::GaussianRandom& ggen, std::size_t n, double stddev = 1.0);

double generateDataPointsA(double x, double noise_stddev);
double generateDataPointsB(double x, double noise_stddev);

std::vector<double> generateDataPointsA(const std::vector<double>& xs, rng::GaussianRandom& ggen, std::size_t n, double noise_stddev);
std::vector<double> generateDataPointsB(const std::vector<double>& xs, rng::GaussianRandom& ggen, std::size_t n, double noise_stddev);

std::function<double(double, const std::vector<double>&)>
make_polynomial(std::size_t order);

std::vector<double>
evaluate_many(const std::function<double(double, const std::vector<double>&)>& f,
              const std::vector<double>& xs,
              const std::vector<double>& theta);

#endif // FUNCUTILS_HPP