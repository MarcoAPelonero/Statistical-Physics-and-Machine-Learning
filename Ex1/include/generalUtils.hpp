#ifndef GENERAL_UTILS_HPP
#define GENERAL_UTILS_HPP

#include <vector>
#include <functional>
#include <cstddef>
#include <string>
#include <fstream>
#include <iostream>
#include "polyFitting.hpp"

struct DataSet {
    std::vector<double> x;
    std::vector<double> dataPointsA;
    std::vector<double> dataPointsB;
};

struct PolynomialSet {
    std::vector<std::function<double(double, const std::vector<double>&)>> polynomials;
    std::vector<std::vector<double>> initial_params;
    std::vector<std::size_t> orders;
};

struct FitResults {
    std::vector<std::vector<double>> fitted_params_A;
    std::vector<std::vector<double>> fitted_params_B;
    std::vector<std::size_t> orders;
};

DataSet readOutputFile(const std::string& filepath);
PolynomialSet initializePoly(const std::vector<std::size_t>& orders, double param_mean = 0.0, double param_stddev = 0.1);
FitResults fitData(const DataSet& dataset, const PolynomialSet& poly_set, const fit::GDOptions& options);
FitResults fitDataSGD(const DataSet& dataset, const PolynomialSet& poly_set, const fit::SGDOptions& options);

void writeComparisonFile(
    const std::string& filename, const DataSet& dataset, const PolynomialSet& poly_set, const std::vector<double>& x_pred, const std::vector<std::pair<std::string, const FitResults*>>& methods, double noise_stddev);

#endif // GENERAL_UTILS_HPP