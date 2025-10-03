#include "generalUtils.hpp"
#include <algorithm>

DataSet readOutputFile(const std::string& filepath) {
    DataSet dataset;
    
    std::ifstream infile(filepath);
    if (!infile.is_open()) {
        std::cerr << "Error: Could not open file " << filepath << std::endl;
        return dataset;  // Return empty dataset
    }
    
    double u, a, b;
    while (infile >> u >> a >> b) {
        dataset.x.push_back(u);
        dataset.dataPointsA.push_back(a);
        dataset.dataPointsB.push_back(b);
    }
    
    infile.close();
    
    std::cout << "Successfully read " << dataset.x.size() << " data points from " << filepath << std::endl;
    
    return dataset;
}

PolynomialSet initializePoly(const std::vector<std::size_t>& orders, double param_mean, double param_stddev) {
    PolynomialSet poly_set;
    rng::GaussianRandom ggen;
    
    poly_set.orders = orders;
    poly_set.polynomials.reserve(orders.size());
    poly_set.initial_params.reserve(orders.size());
    
    std::cout << "Initializing polynomials with orders: ";
    for (std::size_t order : orders) {
        std::cout << order << " ";
        
        auto polynomial = make_polynomial(order);
        poly_set.polynomials.push_back(polynomial);
        
        std::vector<double> initial_params = ggen.next(order + 1, param_mean, param_stddev);
        poly_set.initial_params.push_back(initial_params);
    }
    std::cout << std::endl;
    
    return poly_set;
}

FitResults fitData(const DataSet& dataset, const PolynomialSet& poly_set, const fit::GDOptions& options) {
    FitResults results;
    
    results.orders = poly_set.orders;
    results.fitted_params_A.reserve(poly_set.polynomials.size());
    results.fitted_params_B.reserve(poly_set.polynomials.size());
    
    std::cout << "\nFitting polynomials to dataset A and B..." << std::endl;
    
    for (std::size_t i = 0; i < poly_set.polynomials.size(); ++i) {
        std::size_t order = poly_set.orders[i];
        const auto& polynomial = poly_set.polynomials[i];
        const auto& initial_params = poly_set.initial_params[i];
        
        std::cout << "\nFitting polynomial of order " << order << " to dataPointsA:" << std::endl;
        std::vector<double> fitted_params_A = fit::fit_gd(polynomial, dataset.x, dataset.dataPointsA, initial_params, options);
        results.fitted_params_A.push_back(fitted_params_A);
        
        std::cout << "Fitting polynomial of order " << order << " to dataPointsB:" << std::endl;
        std::vector<double> fitted_params_B = fit::fit_gd(polynomial, dataset.x, dataset.dataPointsB, initial_params, options);
        results.fitted_params_B.push_back(fitted_params_B);
    }
    
    std::cout << "\nFitting completed for all polynomials!" << std::endl;
    
    return results;
}

FitResults fitDataSGD(const DataSet& dataset, const PolynomialSet& poly_set, const fit::SGDOptions& options) {
    FitResults results;

    results.orders = poly_set.orders;
    results.fitted_params_A.reserve(poly_set.polynomials.size());
    results.fitted_params_B.reserve(poly_set.polynomials.size());

    std::cout << "\nFitting polynomials with SGD (batch size=" << options.batch_size << ") to dataset A and B..." << std::endl;

    for (std::size_t i = 0; i < poly_set.polynomials.size(); ++i) {
        std::size_t order = poly_set.orders[i];
        const auto& polynomial = poly_set.polynomials[i];
        const auto& initial_params = poly_set.initial_params[i];

        std::cout << "\n[SGD] Fitting polynomial of order " << order << " to dataPointsA:" << std::endl;
        std::vector<double> fitted_params_A = fit::fit_sgd(polynomial, dataset.x, dataset.dataPointsA, initial_params, options);
        results.fitted_params_A.push_back(std::move(fitted_params_A));

        std::cout << "[SGD] Fitting polynomial of order " << order << " to dataPointsB:" << std::endl;
        std::vector<double> fitted_params_B = fit::fit_sgd(polynomial, dataset.x, dataset.dataPointsB, initial_params, options);
        results.fitted_params_B.push_back(std::move(fitted_params_B));
    }

    std::cout << "\nSGD fitting completed for all polynomials!" << std::endl;

    return results;
}

void writeComparisonFile(
    const std::string& filename,
    const DataSet& dataset,
    const PolynomialSet& poly_set,
    const std::vector<double>& x_pred,
    const std::vector<std::pair<std::string, const FitResults*>>& methods,
    double noise_stddev,
    const TestSet* test_set,
    const CurveSet* test_curve,
    bool include_dataset) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        return;
    }

    bool has_test_set = false;
    std::size_t test_size = 0;
    std::vector<double> x_test;
    std::vector<double> target_test_A;
    std::vector<double> target_test_B;
    if (test_set && !test_set->x.empty()) {
        test_size = test_set->x.size();
        test_size = std::min(test_size, test_set->targetA.size());
        test_size = std::min(test_size, test_set->targetB.size());
        if (test_size > 0) {
            has_test_set = true;
            x_test.assign(test_set->x.begin(), test_set->x.begin() + test_size);
            target_test_A.assign(test_set->targetA.begin(), test_set->targetA.begin() + test_size);
            target_test_B.assign(test_set->targetB.begin(), test_set->targetB.begin() + test_size);
            if (test_set->x.size() != test_size ||
                test_set->targetA.size() != test_size ||
                test_set->targetB.size() != test_size) {
                std::cerr << "Warning: Test set vectors have mismatched sizes. Truncating to "
                          << test_size << " entries." << std::endl;
            }
        }
    }

    bool has_test_curve = false;
    std::size_t curve_size = 0;
    std::vector<double> x_curve;
    std::vector<double> curve_A;
    std::vector<double> curve_B;
    if (test_curve && !test_curve->x.empty()) {
        curve_size = test_curve->x.size();
        curve_size = std::min(curve_size, test_curve->targetA.size());
        curve_size = std::min(curve_size, test_curve->targetB.size());
        if (curve_size > 0) {
            has_test_curve = true;
            x_curve.assign(test_curve->x.begin(), test_curve->x.begin() + curve_size);
            curve_A.assign(test_curve->targetA.begin(), test_curve->targetA.begin() + curve_size);
            curve_B.assign(test_curve->targetB.begin(), test_curve->targetB.begin() + curve_size);
            if (test_curve->x.size() != curve_size ||
                test_curve->targetA.size() != curve_size ||
                test_curve->targetB.size() != curve_size) {
                std::cerr << "Warning: Test curve vectors have mismatched sizes. Truncating to "
                          << curve_size << " entries." << std::endl;
            }
        }
    }

    outfile << "# Dataset comparison output\n";
    outfile << "# Samples: " << dataset.x.size() << "\n";
    outfile << "# Noise stddev: " << noise_stddev << "\n";
    outfile << "# Orders: ";
    for (std::size_t order : poly_set.orders) {
        outfile << order << " ";
    }
    outfile << "\n\n";

    if (include_dataset) {
        outfile << "# Dataset\n";
        outfile << "x dataPointA dataPointB\n";
        for (std::size_t i = 0; i < dataset.x.size(); ++i) {
            outfile << dataset.x[i] << " " << dataset.dataPointsA[i] << " " << dataset.dataPointsB[i] << "\n";
        }
        outfile << "\n";
    }

    if (has_test_set) {
        outfile << "# Test points (x, hiddenA, hiddenB)\n";
        for (std::size_t i = 0; i < test_size; ++i) {
            outfile << x_test[i] << " " << target_test_A[i] << " " << target_test_B[i] << "\n";
        }
        outfile << "\n";
    }

    if (has_test_curve) {
        outfile << "# Test curve (x, hiddenA, hiddenB)\n";
        for (std::size_t i = 0; i < curve_size; ++i) {
            outfile << x_curve[i] << " " << curve_A[i] << " " << curve_B[i] << "\n";
        }
        outfile << "\n";
    }

    for (const auto& method_entry : methods) {
        const std::string& method_name = method_entry.first;
        const FitResults& results = *method_entry.second;

        outfile << "# Method: " << method_name << "\n";
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

        std::vector<std::vector<double>> preds_A;
        std::vector<std::vector<double>> preds_B;
        std::vector<std::vector<double>> test_preds_A;
        std::vector<std::vector<double>> test_preds_B;
        preds_A.reserve(poly_set.polynomials.size());
        preds_B.reserve(poly_set.polynomials.size());
        if (has_test_set) {
            test_preds_A.reserve(poly_set.polynomials.size());
            test_preds_B.reserve(poly_set.polynomials.size());
        }

        for (std::size_t i = 0; i < poly_set.polynomials.size(); ++i) {
            std::vector<double> pred_A(x_pred.size());
            std::vector<double> pred_B(x_pred.size());
            predict(poly_set.polynomials[i], x_pred, results.fitted_params_A[i], pred_A);
            predict(poly_set.polynomials[i], x_pred, results.fitted_params_B[i], pred_B);
            preds_A.push_back(std::move(pred_A));
            preds_B.push_back(std::move(pred_B));

            if (has_test_set) {
                std::vector<double> pred_test_A(x_test.size());
                std::vector<double> pred_test_B(x_test.size());
                predict(poly_set.polynomials[i], x_test, results.fitted_params_A[i], pred_test_A);
                predict(poly_set.polynomials[i], x_test, results.fitted_params_B[i], pred_test_B);
                test_preds_A.push_back(std::move(pred_test_A));
                test_preds_B.push_back(std::move(pred_test_B));
            }
        }

        outfile << "x_pred";
        for (std::size_t order : results.orders) {
            outfile << " fit" << order << "_A";
        }
        for (std::size_t order : results.orders) {
            outfile << " fit" << order << "_B";
        }
        outfile << "\n";

        for (std::size_t idx = 0; idx < x_pred.size(); ++idx) {
            outfile << x_pred[idx];
            for (const auto& pred_vec : preds_A) {
                outfile << " " << pred_vec[idx];
            }
            for (const auto& pred_vec : preds_B) {
                outfile << " " << pred_vec[idx];
            }
            outfile << "\n";
        }

        if (has_test_set) {
            outfile << "\n";
            outfile << "x_test";
            for (std::size_t order : results.orders) {
                outfile << " testfit" << order << "_A";
            }
            for (std::size_t order : results.orders) {
                outfile << " testfit" << order << "_B";
            }
            outfile << "\n";

            for (std::size_t idx = 0; idx < x_test.size(); ++idx) {
                outfile << x_test[idx];
                for (const auto& pred_vec : test_preds_A) {
                    outfile << " " << pred_vec[idx];
                }
                for (const auto& pred_vec : test_preds_B) {
                    outfile << " " << pred_vec[idx];
                }
                outfile << "\n";
            }
        }

        outfile << "\n";
    }

    outfile.close();
    std::cout << "Comparison output written to " << filename << std::endl;
}
