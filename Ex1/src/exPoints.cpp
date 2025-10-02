#include "exPoints.hpp"
#include <algorithm>
#include <string>

void exPointOne() {}

void exPointTwo() {
    std::cout << "\n=== Exercise point n2 ===\n";

    rng::UniformRandom ugen;
    rng::GaussianRandom ggen;

    std::size_t N = 10;

    double min_v = 0.0;
    double max_v = 1.0;
    double std = 0.0;

    std::vector<double> uniform_numbers = ugen.next(N);
    std::vector<double> uniform_numbers_rescaled;
    uniform_numbers_rescaled.reserve(N);
    for (double u : uniform_numbers) uniform_numbers_rescaled.push_back(min_v + (max_v - min_v) * u);

    std::vector<double> dataPointsA = generateDataPointsA(uniform_numbers_rescaled, ggen, N, std);
    std::vector<double> dataPointsB = generateDataPointsB(uniform_numbers_rescaled, ggen, N, std);

    std::ofstream outfile("output.txt");
    for (std::size_t i = 0; i < N; ++i) {
        outfile << uniform_numbers[i] << " " << dataPointsA[i] << " " << dataPointsB[i] << "\n";
    }
    outfile.close();
}

void exPointThree() {
    std::cout << "\n=== Exercise point n3 ===\n";

    DataSet dataset = readOutputFile("output.txt");
    
    std::vector<double> x = dataset.x;
    std::vector<double> dataPointsA = dataset.dataPointsA;
    std::vector<double> dataPointsB = dataset.dataPointsB;

    std::size_t order1 = 1;
    std::size_t order3 = 3;
    std::size_t order10 = 10;

    auto poly1 = make_polynomial(1);
    auto poly3 = make_polynomial(3);
    auto poly10 = make_polynomial(10);

    rng::GaussianRandom ggen;
    
    std::vector<double> params1 = ggen.next(order1 + 1, 0.0, 0.1);
    std::vector<double> params3 = ggen.next(order3 + 1, 0.0, 0.1);
    std::vector<double> params10 = ggen.next(order10 + 1, 0.0, 0.1);

    fit::GDOptions options;
    options.lr = 0.1;
    options.max_epochs = 200000;
    options.l2 = 0.00;
    options.verbose = true;
    options.print_every = 10000;

    std::cout << "\nFitting polynomial of order 1 to dataPointsA:\n";
    std::vector<double> fitted_params1_A = fit::fit_gd(poly1, x, dataPointsA, params1, options);
    std::cout << "\nFitting polynomial of order 3 to dataPointsA:\n";
    std::vector<double> fitted_params3_A = fit::fit_gd(poly3, x, dataPointsA, params3, options);
    std::cout << "\nFitting polynomial of order 10 to dataPointsA:\n";
    std::vector<double> fitted_params10_A = fit::fit_gd(poly10, x, dataPointsA, params10, options);
    std::cout << "\nFitting polynomial of order 1 to dataPointsB:\n";
    std::vector<double> fitted_params1_B = fit::fit_gd(poly1, x, dataPointsB, params1, options);
    std::cout << "\nFitting polynomial of order 3 to dataPointsB:\n";
    std::vector<double> fitted_params3_B = fit::fit_gd(poly3, x, dataPointsB, params3, options);
    std::cout << "\nFitting polynomial of order 10 to dataPointsB:\n";
    std::vector<double> fitted_params10_B = fit::fit_gd(poly10, x, dataPointsB, params10, options);

    std::size_t N_pred = 1000;

    std::vector<double> x_pred(N_pred);
    for (std::size_t i = 0; i < N_pred; ++i) {
        x_pred[i] = static_cast<double>(i) / (N_pred - 1);
    }

    std::vector<double> y_pred1_A(N_pred), y_pred3_A(N_pred), y_pred10_A(N_pred);
    std::vector<double> y_pred1_B(N_pred), y_pred3_B(N_pred), y_pred10_B(N_pred);

    predict(poly1, x_pred, fitted_params1_A, y_pred1_A);
    predict(poly3, x_pred, fitted_params3_A, y_pred3_A);
    predict(poly10, x_pred, fitted_params10_A, y_pred10_A);
    predict(poly1, x_pred, fitted_params1_B, y_pred1_B);
    predict(poly3, x_pred, fitted_params3_B, y_pred3_B);
    predict(poly10, x_pred, fitted_params10_B, y_pred10_B);

    std::ofstream outfile("fitted_output.txt");

    outfile << "# Fitted parameters:\n";
    outfile << "# Order 1 A: ";
    for (const auto& p : fitted_params1_A) outfile << p << " ";
    outfile << "\n# Order 3 A: ";
    for (const auto& p : fitted_params3_A) outfile << p << " ";
    outfile << "\n# Order 10 A: ";
    for (const auto& p : fitted_params10_A) outfile << p << " ";
    outfile << "\n# Order 1 B: ";
    for (const auto& p : fitted_params1_B) outfile << p << " ";
    outfile << "\n# Order 3 B: ";
    for (const auto& p : fitted_params3_B) outfile << p << " ";
    outfile << "\n# Order 10 B: ";
    for (const auto& p : fitted_params10_B) outfile << p << " ";

    outfile << "\nx_pred fit1_A fit3_A fit10_A fit1_B fit3_B fit10_B\n";
    for (std::size_t i = 0; i < x_pred.size(); ++i) {
        outfile << x_pred[i] << " "
                << y_pred1_A[i] << " " << y_pred3_A[i] << " " << y_pred10_A[i] << " "
                << y_pred1_B[i] << " " << y_pred3_B[i] << " " << y_pred10_B[i] << "\n";
    }
    outfile.close();
}

void exPointFour() {
    std::cout << "\n=== Exercise point n4 ===\n";

    DataSet dataset = readOutputFile("output.txt");
    
    std::vector<std::size_t> orders = {1, 3, 10};
    
    PolynomialSet poly_set = initializePoly(orders, 0.0, 0.1);
    
    fit::GDOptions options;
    options.lr = 0.1;
    options.max_epochs = 200000;
    options.l2 = 0.00;
    options.verbose = true;
    options.print_every = 10000;
    
    FitResults results = fitData(dataset, poly_set, options);
    
    rng::UniformRandom ugen;
    std::size_t N_test = 20;

    double min_v = 0.0;
    double max_v = 1.25;

    std::vector<double> uniform_numbers = ugen.next(N_test);
    std::vector<double> uniform_numbers_rescaled;
    uniform_numbers_rescaled.reserve(N_test);
    for (double u : uniform_numbers) uniform_numbers_rescaled.push_back(min_v + (max_v - min_v) * u);

    std::size_t N_plot = 1000;
    std::vector<double> x_plot(N_plot);
    for (std::size_t i = 0; i < N_plot; ++i) {
        x_plot[i] = min_v + (max_v - min_v) * static_cast<double>(i) / (N_plot - 1);
    }

    std::vector<std::vector<double>> y_plot_A, y_plot_B;
    for (std::size_t i = 0; i < poly_set.polynomials.size(); ++i) {
        std::vector<double> pred_A(N_plot), pred_B(N_plot);
        predict(poly_set.polynomials[i], x_plot, results.fitted_params_A[i], pred_A);
        predict(poly_set.polynomials[i], x_plot, results.fitted_params_B[i], pred_B);
        y_plot_A.push_back(pred_A);
        y_plot_B.push_back(pred_B);
    }

    std::ofstream outfile("fitted_output_with_test.txt");

    outfile << "# Test points (u, dataPointA, dataPointB):\n";
    for (std::size_t i = 0; i < N_test; ++i) {
        double u = uniform_numbers_rescaled[i];
        double a = hiddenFunctionA(u);
        double b = hiddenFunctionB(u);
        outfile << u << " " << a << " " << b << "\n";
    }

    outfile << "\nx_pred fit1_A fit3_A fit10_A fit1_B fit3_B fit10_B\n";
    for (std::size_t i = 0; i < x_plot.size(); ++i) {
        outfile << x_plot[i] << " "
                << y_plot_A[0][i] << " " << y_plot_A[1][i] << " " << y_plot_A[2][i] << " "
                << y_plot_B[0][i] << " " << y_plot_B[1][i] << " " << y_plot_B[2][i] << "\n";
    }
    
    outfile.close();
}

void exPointFive() {
    std::cout << "\n=== Exercise point n5 ===\n";
}

void comparison() {
    std::cout << "\n=== Exercise point n5 ===\n";

    const std::size_t N_samples = 20;
    const double min_v = 0.0;
    const double max_v = 1.0;
    const double noise_stddev = 0.00;
    const std::size_t N_plot = 1000;
    const double plot_max = 1.25;

    rng::UniformRandom ugen;
    rng::GaussianRandom ggen;

    std::vector<double> uniform_numbers = ugen.next(N_samples);
    std::vector<double> x;
    x.reserve(N_samples);
    for (double u : uniform_numbers) {
        x.push_back(min_v + (max_v - min_v) * u);
    }

    std::vector<double> dataPointsA = generateDataPointsA(x, ggen, N_samples, noise_stddev);
    std::vector<double> dataPointsB = generateDataPointsB(x, ggen, N_samples, noise_stddev);

    DataSet dataset;
    dataset.x = x;
    dataset.dataPointsA = dataPointsA;
    dataset.dataPointsB = dataPointsB;

    std::vector<std::size_t> orders = {1, 3, 10};
    PolynomialSet poly_set = initializePoly(orders, 0.0, 0.1);

    fit::GDOptions gd_options;
    gd_options.lr = 0.1;
    gd_options.max_epochs = 100000;
    gd_options.l2 = 0.00;
    gd_options.verbose = true;
    gd_options.print_every = 10000;

    FitResults gd_results = fitData(dataset, poly_set, gd_options);

    fit::SGDOptions sgd_full_options;
    sgd_full_options.lr = gd_options.lr;
    sgd_full_options.max_epochs = gd_options.max_epochs;
    sgd_full_options.l2 = gd_options.l2;
    sgd_full_options.verbose = gd_options.verbose;
    sgd_full_options.print_every = gd_options.print_every;
    sgd_full_options.batch_size = N_samples;
    sgd_full_options.drop_last = false;
    sgd_full_options.lr_decay = 1.0;
    sgd_full_options.decay_every = 0;

    FitResults sgd_full_results = fitDataSGD(dataset, poly_set, sgd_full_options);

    fit::SGDOptions sgd_minibatch_options = sgd_full_options;
    sgd_minibatch_options.batch_size = std::min<std::size_t>(2, N_samples);
    if (sgd_minibatch_options.batch_size == 0) {
        sgd_minibatch_options.batch_size = 1;
    }

    FitResults sgd_minibatch_results = fitDataSGD(dataset, poly_set, sgd_minibatch_options);

    std::vector<double> x_plot(N_plot);
    for (std::size_t i = 0; i < N_plot; ++i) {
        x_plot[i] = min_v + (plot_max - min_v) * static_cast<double>(i) / static_cast<double>(N_plot - 1);
    }

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