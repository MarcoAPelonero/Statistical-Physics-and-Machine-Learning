#include "exPoints.hpp"

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

    std::ifstream infile("output.txt");
    std::vector<double> x;
    std::vector<double> dataPointsA;
    std::vector<double> dataPointsB;

    double u, a, b;
    while (infile >> u >> a >> b) {
        x.push_back(u);
        dataPointsA.push_back(a);
        dataPointsB.push_back(b);
    }

    infile.close();

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
    options.lr = 0.5;
    options.max_epochs = 5000;
    options.l2 = 0.00;
    options.verbose = true;
    options.print_every = 100;

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

    // Now using the fitted parameters to make predictions with x now being 1000 points between 0 and 1
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
    outfile << "x_pred fit1_A fit3_A fit10_A fit1_B fit3_B fit10_B\n";
    for (std::size_t i = 0; i < x_pred.size(); ++i) {
        outfile << x_pred[i] << " "
                << y_pred1_A[i] << " " << y_pred3_A[i] << " " << y_pred10_A[i] << " "
                << y_pred1_B[i] << " " << y_pred3_B[i] << " " << y_pred10_B[i] << "\n";
    }
    outfile.close();
}