#ifndef EXERCISE_UTILS_HPP
#define EXERCISE_UTILS_HPP

#include <vector>
#include <string>
#include "generalUtils.hpp"
#include "funcUtils.hpp"
#include "rng.hpp"

namespace exercise {

struct Range {
    double min;
    double max;
};

struct SampledInputs {
    std::vector<double> raw;
    std::vector<double> scaled;
};

SampledInputs sampleInputs(rng::UniformRandom& ugen, std::size_t count, Range range);

std::vector<double> scaleSamples(const std::vector<double>& samples, Range range);

DataSet buildDataset(const std::vector<double>& xs, rng::GaussianRandom& ggen, double noise_stddev);
DataSet buildDataset(const SampledInputs& inputs, rng::GaussianRandom& ggen, double noise_stddev);

std::vector<double> linspace(Range range, std::size_t count);

TestSet buildTestSet(const std::vector<double>& xs);
TestSet sampleTestSet(rng::UniformRandom& ugen, std::size_t count, Range range);

CurveSet buildCurveSet(const std::vector<double>& xs);

struct PredictionSet {
    std::vector<std::vector<double>> A;
    std::vector<std::vector<double>> B;
};

PredictionSet computePredictions(const PolynomialSet& poly_set,
                                 const FitResults& results,
                                 const std::vector<double>& xs);

fit::GDOptions makeGDOptions(double lr,
                             std::size_t max_epochs,
                             double l2,
                             bool verbose,
                             std::size_t print_every);

fit::SGDOptions makeSGDOptions(double lr,
                               std::size_t max_epochs,
                               double l2,
                               bool verbose,
                               std::size_t print_every,
                               std::size_t batch_size,
                               bool drop_last = false,
                               double lr_decay = 1.0,
                               std::size_t decay_every = 0);

void writeDataset(const std::string& filename,
                  const std::vector<double>& values_to_write,
                  const DataSet& dataset);

void writeFittedSummary(const std::string& filename,
                        const FitResults& results,
                        const std::vector<double>& x_pred,
                        const PredictionSet& predictions);

void writeFittedSummaryWithTest(const std::string& filename,
                                const TestSet& test_set,
                                const FitResults& results,
                                const std::vector<double>& x_pred,
                                const PredictionSet& predictions);

} // namespace exercise

#endif // EXERCISE_UTILS_HPP

