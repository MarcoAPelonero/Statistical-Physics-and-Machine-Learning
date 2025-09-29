#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <string>
#ifdef _WIN32
#include <direct.h>
#endif
#include <sys/stat.h>
#include <sys/types.h>
#include "funcUtils.hpp"

using std::cout;
using std::endl;

// Function to create directory if it doesn't exist
void create_directory(const std::string& path) {
#ifdef _WIN32
    _mkdir(path.c_str());
#else
    mkdir(path.c_str(), 0755);
#endif
}

struct Stats {
    double mean;
    double stddev;
    double min_val;
    double max_val;
};

Stats compute_stats(const std::vector<double>& x) {
    if (x.empty()) return {0.0, 0.0, 0.0, 0.0};
    
    double sum = std::accumulate(x.begin(), x.end(), 0.0);
    double mean = sum / static_cast<double>(x.size());
    
    double variance = 0.0;
    for (double xi : x) {
        variance += (xi - mean) * (xi - mean);
    }
    variance /= static_cast<double>(x.size());
    
    auto minmax = std::minmax_element(x.begin(), x.end());
    
    return {mean, std::sqrt(variance), *minmax.first, *minmax.second};
}

void test_hidden_functions() {
    cout << "\n=== Testing Hidden Functions ===\n";
    
    std::vector<double> test_x = {-2.0, -1.0, 0.0, 1.0, 2.0};
    
    cout << "Single value tests:\n";
    cout << std::fixed << std::setprecision(6);
    cout << "x\t\thiddenFunctionA(x)\thiddenFunctionB(x)\n";
    for (double x : test_x) {
        double ya = hiddenFunctionA(x);
        double yb = hiddenFunctionB(x);
        cout << x << "\t\t" << ya << "\t\t" << yb << "\n";
    }
    
    std::vector<double> ya_vec = hiddenFunctionA(test_x);
    std::vector<double> yb_vec = hiddenFunctionB(test_x);
    
    cout << "\nVector function consistency check:\n";
    bool consistent = true;
    for (std::size_t i = 0; i < test_x.size(); ++i) {
        double single_a = hiddenFunctionA(test_x[i]);
        double single_b = hiddenFunctionB(test_x[i]);
        if (std::abs(ya_vec[i] - single_a) > 1e-10 || 
            std::abs(yb_vec[i] - single_b) > 1e-10) {
            consistent = false;
            cout << "Inconsistency at x=" << test_x[i] << "\n";
        }
    }
    if (consistent) {
        cout << "✓ Vector functions consistent with single value functions\n";
    }
    
    std::vector<double> ya_inplace, yb_inplace;
    hiddenFunctionA(test_x, ya_inplace);
    hiddenFunctionB(test_x, yb_inplace);
    
    cout << "In-place function consistency check:\n";
    consistent = true;
    for (std::size_t i = 0; i < test_x.size(); ++i) {
        if (std::abs(ya_inplace[i] - ya_vec[i]) > 1e-10 || 
            std::abs(yb_inplace[i] - yb_vec[i]) > 1e-10) {
            consistent = false;
            cout << "Inconsistency at x=" << test_x[i] << "\n";
        }
    }
    if (consistent) {
        cout << "✓ In-place functions consistent with vector functions\n";
    }
}

void test_noise_generation() {
    cout << "\n=== Testing Noise Generation ===\n";
    
    rng::GaussianRandom ggen(12345); 
    cout << "Single noise values (stddev=1.0):\n";
    for (int i = 0; i < 5; ++i) {
        double noise = generateNoise(ggen, 1.0);
        cout << "  " << noise << "\n";
    }
    
    const std::size_t N = 10000;
    std::vector<double> noise_vec = generateNoise(ggen, N, 2.5);
    Stats stats = compute_stats(noise_vec);
    
    cout << "\nNoise statistics (N=" << N << ", stddev=2.5):\n";
    cout << "  Mean: " << stats.mean << " (expected: ~0.0)\n";
    cout << "  Stddev: " << stats.stddev << " (expected: ~2.5)\n";
    cout << "  Min: " << stats.min_val << "\n";
    cout << "  Max: " << stats.max_val << "\n";
}

void test_data_generation() {
    cout << "\n=== Testing Data Point Generation ===\n";
    
    double x_test = 1.5;
    double noise_std = 0.1;
    
    cout << "Single data point generation at x=" << x_test << " with noise_stddev=" << noise_std << ":\n";
    cout << "Pure functions: A=" << hiddenFunctionA(x_test) << ", B=" << hiddenFunctionB(x_test) << "\n";
    
    cout << "With noise (5 samples each):\n";
    for (int i = 0; i < 5; ++i) {
        double noisy_a = generateDataPointsA(x_test, noise_std);
        double noisy_b = generateDataPointsB(x_test, noise_std);
        cout << "  A: " << noisy_a << ", B: " << noisy_b << "\n";
    }
    
    std::vector<double> x_vec = {-1.0, 0.0, 1.0, 2.0};
    std::vector<double> noisy_a_vec = generateDataPointsA(x_vec, x_vec.size(), noise_std);
    std::vector<double> noisy_b_vec = generateDataPointsB(x_vec, x_vec.size(), noise_std);
    
    cout << "\nVector data generation:\n";
    cout << "x\t\tNoisy A\t\tNoisy B\n";
    for (std::size_t i = 0; i < x_vec.size(); ++i) {
        cout << x_vec[i] << "\t\t" << noisy_a_vec[i] << "\t\t" << noisy_b_vec[i] << "\n";
    }
}

void test_polynomial_functions() {
    cout << "\n=== Testing Polynomial Functions ===\n";
    
    // Test polynomial creation and evaluation
    std::vector<std::size_t> orders = {1, 2, 3, 5};
    double x_test = 2.0;
    
    for (std::size_t order : orders) {
        auto poly = make_polynomial(order);
        
        std::vector<double> theta(order + 1, 1.0);
        
        double result = poly(x_test, theta);
        
        double expected = 0.0;
        double x_power = 1.0;
        for (std::size_t i = 0; i <= order; ++i) {
            expected += x_power;
            x_power *= x_test;
        }
        
        cout << "Order " << order << " polynomial at x=" << x_test << " with theta=[";
        for (std::size_t i = 0; i < theta.size(); ++i) {
            cout << theta[i];
            if (i < theta.size() - 1) cout << ",";
        }
        cout << "]: " << result << " (expected: " << expected << ")\n";
        
        if (std::abs(result - expected) < 1e-10) {
            cout << "  ✓ Correct\n";
        } else {
            cout << "  ✗ Error!\n";
        }
    }
    
    std::vector<double> x_vec = {0.0, 1.0, 2.0, 3.0};
    std::vector<double> theta = {1.0, 2.0, 1.0}; 
    auto quadratic = make_polynomial(2);
    
    std::vector<double> results = evaluate_many(quadratic, x_vec, theta);
    
    cout << "\nQuadratic polynomial 1 + 2x + x^2 evaluation:\n";
    cout << "x\t\ty\n";
    for (std::size_t i = 0; i < x_vec.size(); ++i) {
        double x = x_vec[i];
        double expected = 1.0 + 2.0*x + x*x;
        cout << x << "\t\t" << results[i] << " (expected: " << expected << ")\n";
    }
}

void generate_function_data_files() {
    cout << "\n=== Generating Data Files for Plotting ===\n";
    
    // Create output directory
    const std::string data_dir = "testFuncUtilsData";
    create_directory(data_dir);
    
    // Generate data for hidden functions
    std::ofstream func_file(data_dir + "/hidden_functions.dat");
    func_file << "# x hiddenA hiddenB\n";
    
    const double x_min = -2.0, x_max = 2.0;
    const int n_points = 200;
    const double dx = (x_max - x_min) / (n_points - 1);
    
    for (int i = 0; i < n_points; ++i) {
        double x = x_min + i * dx;
        double ya = hiddenFunctionA(x);
        double yb = hiddenFunctionB(x);
        func_file << x << " " << ya << " " << yb << "\n";
    }
    func_file.close();
    cout << "✓ Generated " << data_dir << "/hidden_functions.dat\n";
    
    // Generate noisy data
    std::ofstream noisy_file(data_dir + "/noisy_data.dat");
    noisy_file << "# x noisyA_01 noisyA_05 noisyB_01 noisyB_05\n";
    
    const int n_noisy = 50;
    const double dx_noisy = (x_max - x_min) / (n_noisy - 1);
    
    for (int i = 0; i < n_noisy; ++i) {
        double x = x_min + i * dx_noisy;
        double noisy_a_01 = generateDataPointsA(x, 0.1);
        double noisy_a_05 = generateDataPointsA(x, 0.5);
        double noisy_b_01 = generateDataPointsB(x, 0.1);
        double noisy_b_05 = generateDataPointsB(x, 0.5);
        
        noisy_file << x << " " << noisy_a_01 << " " << noisy_a_05 << " " 
                   << noisy_b_01 << " " << noisy_b_05 << "\n";
    }
    noisy_file.close();
    cout << "✓ Generated " << data_dir << "/noisy_data.dat\n";
    
    // Generate polynomial data
    std::ofstream poly_file(data_dir + "/polynomial_data.dat");
    poly_file << "# x linear quadratic cubic quintic\n";
    
    auto linear = make_polynomial(1);
    auto quadratic = make_polynomial(2);
    auto cubic = make_polynomial(3);
    auto quintic = make_polynomial(5);
    
    std::vector<double> theta_linear = {1.0, 2.0};           // 1 + 2x
    std::vector<double> theta_quad = {1.0, 0.0, 1.0};       // 1 + x^2
    std::vector<double> theta_cubic = {0.0, 1.0, 0.0, 1.0}; // x + x^3
    std::vector<double> theta_quintic = {0.0, 1.0, 0.0, -0.5, 0.0, 0.1}; // x - 0.5x^3 + 0.1x^5
    
    for (int i = 0; i < n_points; ++i) {
        double x = x_min + i * dx;
        double y_lin = linear(x, theta_linear);
        double y_quad = quadratic(x, theta_quad);
        double y_cubic = cubic(x, theta_cubic);
        double y_quin = quintic(x, theta_quintic);
        
        poly_file << x << " " << y_lin << " " << y_quad << " " 
                  << y_cubic << " " << y_quin << "\n";
    }
    poly_file.close();
    cout << "✓ Generated " << data_dir << "/polynomial_data.dat\n";
    
    // Generate noise statistics
    std::ofstream noise_file(data_dir + "/noise_stats.dat");
    noise_file << "# stddev mean measured_stddev\n";
    
    std::vector<double> stddevs = {0.1, 0.5, 1.0, 2.0, 5.0};
    rng::GaussianRandom ggen(42);
    
    for (double stddev : stddevs) {
        std::vector<double> noise = generateNoise(ggen, 10000, stddev);
        Stats stats = compute_stats(noise);
        noise_file << stddev << " " << stats.mean << " " << stats.stddev << "\n";
    }
    noise_file.close();
    cout << "✓ Generated " << data_dir << "/noise_stats.dat\n";
}

int main() {
    cout << "=== Comprehensive funcUtils Testing ===\n";
    
    test_hidden_functions();
    test_noise_generation();
    test_data_generation();
    test_polynomial_functions();
    generate_function_data_files();
    
    cout << "\n=== All Tests Completed ===\n";
    cout << "Data files generated in testFuncUtilsData/ folder. Run funcUtilsPlotter.py to visualize results.\n";
    
    return 0;
}