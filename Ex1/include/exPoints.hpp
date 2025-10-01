#ifndef EXPOINTS_HPP
#define EXPOINTS_HPP

#include <iostream>
#include <functional>
#include <fstream>
#include <cstdint>
#include "polyFitting.hpp"
#include "funcUtils.hpp"
#include "rng.hpp"

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

void exPointOne();
void exPointTwo();
void exPointThree();
void exPointFour();
void exPointFive();

DataSet readOutputFile(const std::string& filepath);
PolynomialSet initializePoly(const std::vector<std::size_t>& orders, double param_mean = 0.0, double param_stddev = 0.1);
FitResults fitData(const DataSet& dataset, const PolynomialSet& poly_set, const fit::GDOptions& options);

#endif // EXPOINTS_HPP