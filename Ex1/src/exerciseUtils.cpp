#include "exerciseUtils.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include "polyFitting.hpp"

namespace {

void writePredictionTable(std::ofstream& outfile,
                          const FitResults& results,
                          const std::vector<double>& x_pred,
                          const exercise::PredictionSet& predictions)
{
    if (!outfile.is_open()) {
        return;
    }

    outfile << "x_pred";
    for (std::size_t order : results.orders) {
        outfile << " fit" << order << "_A";
    }
    for (std::size_t order : results.orders) {
        outfile << " fit" << order << "_B";
    }
    outfile << "\n";

    const std::size_t rows = x_pred.size();
    for (std::size_t idx = 0; idx < rows; ++idx) {
        outfile << x_pred[idx];
        for (const auto& pred_vec : predictions.A) {
            outfile << " " << pred_vec[idx];
        }
        for (const auto& pred_vec : predictions.B) {
            outfile << " " << pred_vec[idx];
        }
        outfile << "\n";
    }
}

} // namespace

namespace exercise {

SampledInputs sampleInputs(rng::UniformRandom& ugen, std::size_t count, Range range)
{
    SampledInputs inputs;
    inputs.raw = ugen.next(count);
    inputs.scaled = scaleSamples(inputs.raw, range);
    return inputs;
}

std::vector<double> scaleSamples(const std::vector<double>& samples, Range range)
{
    std::vector<double> scaled;
    scaled.reserve(samples.size());
    const double span = range.max - range.min;
    for (double value : samples) {
        scaled.push_back(range.min + span * value);
    }
    return scaled;
}

DataSet buildDataset(const std::vector<double>& xs, rng::GaussianRandom& ggen, double noise_stddev)
{
    DataSet dataset;
    dataset.x = xs;
    const std::size_t n = xs.size();
    dataset.dataPointsA = generateDataPointsA(xs, ggen, n, noise_stddev);
    dataset.dataPointsB = generateDataPointsB(xs, ggen, n, noise_stddev);
    return dataset;
}

DataSet buildDataset(const SampledInputs& inputs, rng::GaussianRandom& ggen, double noise_stddev)
{
    return buildDataset(inputs.scaled, ggen, noise_stddev);
}

std::vector<double> linspace(Range range, std::size_t count)
{
    std::vector<double> values;
    if (count == 0) {
        return values;
    }
    values.resize(count);
    if (count == 1) {
        values[0] = range.min;
        return values;
    }
    const double span = range.max - range.min;
    for (std::size_t i = 0; i < count; ++i) {
        values[i] = range.min + span * static_cast<double>(i) / static_cast<double>(count - 1);
    }
    return values;
}

TestSet buildTestSet(const std::vector<double>& xs)
{
    TestSet test_set;
    test_set.x = xs;
    test_set.targetA = hiddenFunctionA(xs);
    test_set.targetB = hiddenFunctionB(xs);
    return test_set;
}

TestSet sampleTestSet(rng::UniformRandom& ugen, std::size_t count, Range range)
{
    SampledInputs inputs = sampleInputs(ugen, count, range);
    return buildTestSet(inputs.scaled);
}

CurveSet buildCurveSet(const std::vector<double>& xs)
{
    CurveSet curve_set;
    curve_set.x = xs;
    curve_set.targetA = hiddenFunctionA(xs);
    curve_set.targetB = hiddenFunctionB(xs);
    return curve_set;
}

PredictionSet computePredictions(const PolynomialSet& poly_set,
                                 const FitResults& results,
                                 const std::vector<double>& xs)
{
    PredictionSet prediction_set;
    const std::size_t n_polys = poly_set.polynomials.size();
    prediction_set.A.reserve(n_polys);
    prediction_set.B.reserve(n_polys);

    for (std::size_t i = 0; i < n_polys; ++i) {
        std::vector<double> pred_A(xs.size());
        std::vector<double> pred_B(xs.size());
        predict(poly_set.polynomials[i], xs, results.fitted_params_A[i], pred_A);
        predict(poly_set.polynomials[i], xs, results.fitted_params_B[i], pred_B);
        prediction_set.A.push_back(std::move(pred_A));
        prediction_set.B.push_back(std::move(pred_B));
    }

    return prediction_set;
}

fit::GDOptions makeGDOptions(double lr,
                             std::size_t max_epochs,
                             double l2,
                             bool verbose,
                             std::size_t print_every)
{
    fit::GDOptions options;
    options.lr = lr;
    options.max_epochs = max_epochs;
    options.l2 = l2;
    options.verbose = verbose;
    options.print_every = print_every;
    return options;
}

fit::SGDOptions makeSGDOptions(double lr,
                               std::size_t max_epochs,
                               double l2,
                               bool verbose,
                               std::size_t print_every,
                               std::size_t batch_size,
                               bool drop_last,
                               double lr_decay,
                               std::size_t decay_every)
{
    fit::SGDOptions options;
    options.lr = lr;
    options.max_epochs = max_epochs;
    options.l2 = l2;
    options.verbose = verbose;
    options.print_every = print_every;
    options.batch_size = batch_size;
    options.drop_last = drop_last;
    options.lr_decay = lr_decay;
    options.decay_every = decay_every;
    return options;
}

void writeDataset(const std::string& filename,
                  const std::vector<double>& values_to_write,
                  const DataSet& dataset)
{
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        return;
    }

    const std::size_t n = std::min({values_to_write.size(), dataset.dataPointsA.size(), dataset.dataPointsB.size()});
    for (std::size_t i = 0; i < n; ++i) {
        outfile << values_to_write[i] << " " << dataset.dataPointsA[i] << " " << dataset.dataPointsB[i] << "\n";
    }
}

void writeFittedSummary(const std::string& filename,
                        const FitResults& results,
                        const std::vector<double>& x_pred,
                        const PredictionSet& predictions)
{
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        return;
    }

    outfile << "# Fitted parameters:\n";
    for (std::size_t i = 0; i < results.orders.size(); ++i) {
        outfile << "# Order " << results.orders[i] << " A: ";
        for (double param : results.fitted_params_A[i]) {
            outfile << param << " ";
        }
        outfile << "\n";
    }
    for (std::size_t i = 0; i < results.orders.size(); ++i) {
        outfile << "# Order " << results.orders[i] << " B: ";
        for (double param : results.fitted_params_B[i]) {
            outfile << param << " ";
        }
        outfile << "\n";
    }

    outfile << "\n";
    writePredictionTable(outfile, results, x_pred, predictions);
}

void writeFittedSummaryWithTest(const std::string& filename,
                                const TestSet& test_set,
                                const FitResults& results,
                                const std::vector<double>& x_pred,
                                const PredictionSet& predictions)
{
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        return;
    }

    outfile << "# Test points (u, dataPointA, dataPointB):\n";
    const std::size_t test_size = std::min({test_set.x.size(), test_set.targetA.size(), test_set.targetB.size()});
    for (std::size_t i = 0; i < test_size; ++i) {
        outfile << test_set.x[i] << " " << test_set.targetA[i] << " " << test_set.targetB[i] << "\n";
    }

    outfile << "\n";
    writePredictionTable(outfile, results, x_pred, predictions);
}

} // namespace exercise
