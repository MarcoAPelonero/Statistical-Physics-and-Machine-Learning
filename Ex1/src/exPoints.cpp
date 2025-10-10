#include "exPoints.hpp"
#include "exerciseUtils.hpp"
#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace {
using exercise::Range;

struct GeneratedDataset {
    exercise::SampledInputs inputs;
    DataSet dataset;
};

GeneratedDataset generateDataset(std::size_t count,
                                 Range range,
                                 double noise_stddev,
                                 rng::UniformRandom& ugen,
                                 rng::GaussianRandom& ggen)
{
    GeneratedDataset result;
    result.inputs = exercise::sampleInputs(ugen, count, range);
    result.dataset = exercise::buildDataset(result.inputs, ggen, noise_stddev);
    return result;
}

std::vector<std::size_t> defaultOrders()
{
    return {1, 3, 10};
}

const Range unit_range{0.0, 1.0};

} // namespace

void exPointOne() {}

void exPointTwo()
{
    std::cout << "\n=== Exercise point n2 ===\n";

    rng::UniformRandom ugen;
    rng::GaussianRandom ggen;

    const std::size_t sample_count = 10;
    const double noise_stddev = 0.0;

    GeneratedDataset generated = generateDataset(sample_count, unit_range, noise_stddev, ugen, ggen);
    exercise::writeDataset("output.txt", generated.inputs.raw, generated.dataset);
}

void exPointThree()
{
    std::cout << "\n=== Exercise point n3 ===\n";

    DataSet dataset = readOutputFile("output.txt");
    if (dataset.x.empty()) {
        std::cerr << "Dataset is empty. Did you run exercise point 2 first?" << std::endl;
        return;
    }

    PolynomialSet poly_set = initializePoly(defaultOrders(), 0.0, 0.1);

    fit::GDOptions gd_options = exercise::makeGDOptions(0.1, 200000, 0.0, true, 10000);
    FitResults results = fitData(dataset, poly_set, gd_options);

    std::vector<double> x_pred = exercise::linspace(unit_range, 1000);
    exercise::PredictionSet predictions = exercise::computePredictions(poly_set, results, x_pred);

    exercise::writeFittedSummary("fitted_output.txt", results, x_pred, predictions);
}

void exPointFour()
{
    std::cout << "\n=== Exercise point n4 ===\n";

    DataSet dataset = readOutputFile("output.txt");
    if (dataset.x.empty()) {
        std::cerr << "Dataset is empty. Did you run exercise point 2 first?" << std::endl;
        return;
    }

    PolynomialSet poly_set = initializePoly(defaultOrders(), 0.0, 0.1);

    fit::GDOptions gd_options = exercise::makeGDOptions(0.1, 20000, 0.0, true, 10000);
    FitResults results = fitData(dataset, poly_set, gd_options);

    rng::UniformRandom ugen;
    const Range plot_range{0.0, 1.25};

    TestSet test_set = exercise::sampleTestSet(ugen, 20, plot_range);
    std::vector<double> x_plot = exercise::linspace(plot_range, 1000);
    exercise::PredictionSet predictions = exercise::computePredictions(poly_set, results, x_plot);

    exercise::writeFittedSummaryWithTest("fitted_output_with_test.txt", test_set, results, x_plot, predictions);
}

void exPointFive()
{
    std::cout << "\n=== Exercise point n5 ===\n";
}

void exPointSix()
{
    std::cout << "\n=== Exercise point n6 ===\n";
}

void exPointSeven()
{
    std::cout << "\n=== Exercise point n7 ===\n";

    const std::size_t N_samples = 100;
    const double noise_stddev = 1.0;
    const std::size_t N_plot = 1000;
    const std::size_t N_test = 20;
    const double plot_max = 1.25;

    rng::UniformRandom ugen;
    rng::GaussianRandom ggen;

    GeneratedDataset generated = generateDataset(N_samples, unit_range, noise_stddev, ugen, ggen);
    DataSet dataset = std::move(generated.dataset);

    std::vector<std::size_t> orders = defaultOrders();
    PolynomialSet poly_set = initializePoly(orders, 0.0, 0.2);

    fit::GDOptions gd_options = exercise::makeGDOptions(0.2, 10000, 0.0, true, 2000);
    FitResults gd_results = fitData(dataset, poly_set, gd_options);

    fit::SGDOptions sgd_full_options = exercise::makeSGDOptions(gd_options.lr,
                                                                gd_options.max_epochs,
                                                                gd_options.l2,
                                                                gd_options.verbose,
                                                                gd_options.print_every,
                                                                10,
                                                                false,
                                                                0.5,
                                                                2000);
    FitResults sgd_full_results = fitDataSGD(dataset, poly_set, sgd_full_options);

    const Range plot_range{0.0, plot_max};
    std::vector<double> x_plot = exercise::linspace(plot_range, N_plot);
    TestSet test_set = exercise::sampleTestSet(ugen, N_test, plot_range);
    CurveSet curve_set = exercise::buildCurveSet(x_plot);

    std::vector<std::pair<std::string, const FitResults*>> methods;
    methods.emplace_back("GD", &gd_results);
    methods.emplace_back("SGD batch=" + std::to_string(sgd_full_options.batch_size), &sgd_full_results);

    const std::string comparison_filename = "exercise7_comparison.txt";
    writeComparisonFile(comparison_filename, dataset, poly_set, x_plot, methods, noise_stddev, &test_set, &curve_set, false);

    std::cout << "Exercise 7 comparison data written to " << comparison_filename << std::endl;
}

void exPointEight()
{
    std::cout << "\n=== Exercise point n8 ===\n";
}

void exPointNine()
{
    std::cout << "\n=== Exercise point n9 ===\n";

    const std::size_t N_samples = 10000;
    const double noise_stddev = 1.0;
    const std::size_t N_plot = 1000;
    const std::size_t N_test = 20;
    const double plot_max = 1.25;

    rng::UniformRandom ugen;
    rng::GaussianRandom ggen;

    GeneratedDataset generated = generateDataset(N_samples, unit_range, noise_stddev, ugen, ggen);
    DataSet dataset = std::move(generated.dataset);

    std::vector<std::size_t> orders = defaultOrders();
    PolynomialSet poly_set = initializePoly(orders, 0.0, 0.2);

    fit::GDOptions gd_options = exercise::makeGDOptions(0.2, 10000, 0.0, true, 2000);
    FitResults gd_results = fitData(dataset, poly_set, gd_options);

    fit::SGDOptions sgd_full_options = exercise::makeSGDOptions(gd_options.lr,
                                                                gd_options.max_epochs,
                                                                gd_options.l2,
                                                                gd_options.verbose,
                                                                gd_options.print_every,
                                                                100,
                                                                false,
                                                                1.0,
                                                                0);
    FitResults sgd_full_results = fitDataSGD(dataset, poly_set, sgd_full_options);

    const Range plot_range{0.0, plot_max};
    std::vector<double> x_plot = exercise::linspace(plot_range, N_plot);
    TestSet test_set = exercise::sampleTestSet(ugen, N_test, plot_range);
    CurveSet curve_set = exercise::buildCurveSet(x_plot);

    std::vector<std::pair<std::string, const FitResults*>> methods;
    methods.emplace_back("GD", &gd_results);
    methods.emplace_back("SGD batch=" + std::to_string(sgd_full_options.batch_size), &sgd_full_results);

    const std::string comparison_filename = "exercise9_comparison.txt";
    writeComparisonFile(comparison_filename, dataset, poly_set, x_plot, methods, noise_stddev, &test_set, &curve_set, false);

    std::cout << "Exercise 9 comparison data written to " << comparison_filename << std::endl;
}

void comparison()
{
    std::cout << "\n=== Comparison ===\n";

    const std::size_t N_samples = 20;
    const double min_v = 0.0;
    const double max_v = 1.0;
    const double noise_stddev = 0.0;
    const std::size_t N_plot = 1000;
    const double plot_max = 1.25;

    rng::UniformRandom ugen;
    rng::GaussianRandom ggen;

    const Range sample_range{min_v, max_v};
    GeneratedDataset generated = generateDataset(N_samples, sample_range, noise_stddev, ugen, ggen);
    DataSet dataset = std::move(generated.dataset);

    std::vector<std::size_t> orders = defaultOrders();
    PolynomialSet poly_set = initializePoly(orders, 0.0, 0.1);

    fit::GDOptions gd_options = exercise::makeGDOptions(0.1, 100000, 0.0, true, 10000);
    FitResults gd_results = fitData(dataset, poly_set, gd_options);

    fit::SGDOptions sgd_full_options = exercise::makeSGDOptions(gd_options.lr,
                                                                gd_options.max_epochs,
                                                                gd_options.l2,
                                                                gd_options.verbose,
                                                                gd_options.print_every,
                                                                N_samples);
    FitResults sgd_full_results = fitDataSGD(dataset, poly_set, sgd_full_options);

    std::size_t minibatch_size = std::min<std::size_t>(2, N_samples);
    if (minibatch_size == 0) {
        minibatch_size = 1;
    }
    fit::SGDOptions sgd_minibatch_options = exercise::makeSGDOptions(gd_options.lr,
                                                                     gd_options.max_epochs,
                                                                     gd_options.l2,
                                                                     gd_options.verbose,
                                                                     gd_options.print_every,
                                                                     minibatch_size);
    FitResults sgd_minibatch_results = fitDataSGD(dataset, poly_set, sgd_minibatch_options);

    const Range plot_range{min_v, plot_max};
    std::vector<double> x_plot = exercise::linspace(plot_range, N_plot);

    std::vector<std::pair<std::string, const FitResults*>> full_batch_methods;
    full_batch_methods.emplace_back("GD", &gd_results);
    full_batch_methods.emplace_back("SGD batch=" + std::to_string(N_samples), &sgd_full_results);
    writeComparisonFile("gd_vs_sgd_fullbatch.txt", dataset, poly_set, x_plot, full_batch_methods, noise_stddev);

    std::vector<std::pair<std::string, const FitResults*>> minibatch_methods;
    minibatch_methods.emplace_back("GD", &gd_results);
    minibatch_methods.emplace_back("SGD batch=2", &sgd_minibatch_results);
    writeComparisonFile("gd_vs_sgd_minibatch2.txt", dataset, poly_set, x_plot, minibatch_methods, noise_stddev);

    std::cout << "Generated GD vs SGD comparison files (full batch and mini-batch)" << std::endl;
}
