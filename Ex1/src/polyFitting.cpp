#include "polyFitting.hpp"

void predict(
    const std::function<double(double, const std::vector<double>&)>& f,
    const std::vector<double>& x,
    const std::vector<double>& params,
    std::vector<double>& y_pred) {
    const std::size_t n = x.size();
    for (std::size_t i = 0; i < n; ++i) {
        y_pred[i] = f(x[i], params);
    }
}

inline static double compute_mse(
    const std::vector<double>& y,
    const std::vector<double>& y_pred) {
    const std::size_t n = y.size();
    double sum_q = 0.0;
    double residual = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        residual = y_pred[i] - y[i];
        sum_q += residual * residual;
    }
    double mse = sum_q / static_cast<double>(n);
    return mse;
}

inline static double compute_gradient_component(
    const std::function<double(double, const std::vector<double>&)>& f,
    const std::vector<double>& x,
    const std::vector<double>& y,
    const std::vector<double>& params,
    std::size_t param_index,
    double fd_eps) {
    const std::size_t n = x.size();

    std::vector<double> params_plus = params;
    std::vector<double> params_minus = params;
    params_plus[param_index] += fd_eps;
    params_minus[param_index] -= fd_eps;

    std::vector<double> y_pred_plus(n);
    std::vector<double> y_pred_minus(n);
    predict(f, x, params_plus, y_pred_plus);
    predict(f, x, params_minus, y_pred_minus);

    double mse_plus = compute_mse(y, y_pred_plus);
    double mse_minus = compute_mse(y, y_pred_minus);

    double gradient_component = (mse_plus - mse_minus) / (2.0 * fd_eps);
    return gradient_component;
}

inline static void compute_gradient(
    const std::function<double(double, const std::vector<double>&)>& f,
    const std::vector<double>& x,
    const std::vector<double>& y,
    const std::vector<double>& params,
    std::vector<double>& gradient,
    double fd_eps) {
    const std::size_t p = params.size();
    for (std::size_t j = 0; j < p; ++j) {
        gradient[j] = compute_gradient_component(f, x, y, params, j, fd_eps);
    }
}

namespace fit {

void epoch(
    const std::function<double(double, const std::vector<double>&)>& f,
    const std::vector<double>& x,
    const std::vector<double>& y,
    std::vector<double>& params,
    double lr,
    double l2,
    double fd_eps) {
    const std::size_t p = params.size();
    std::vector<double> gradient(p, 0.0);
    compute_gradient(f, x, y, params, gradient, fd_eps);
    for (std::size_t j = 0; j < p; ++j) {
        params[j] -= lr * (gradient[j] + l2 * params[j]);
    }
}

std::vector<double> fit_gd(
    const std::function<double(double, const std::vector<double>&)>& f,
    const std::vector<double>& x,
    const std::vector<double>& y,
    std::vector<double> params0,
    const GDOptions& options,
    double fd_eps) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("x and y must have the same size");
    }
    const std::size_t n = x.size();
    const std::size_t p = params0.size();
    if (n == 0 || p == 0) {
        throw std::invalid_argument("x, y, and params0 must be non-empty");
    }

    std::vector<double> params = params0;
    for (std::size_t epoch_idx = 0; epoch_idx < options.max_epochs; ++epoch_idx) {
        epoch(f, x, y, params, options.lr, options.l2, fd_eps);
        if (options.verbose && (epoch_idx % options.print_every == 0 || epoch_idx == options.max_epochs - 1)) {
            std::vector<double> y_pred(n);
            predict(f, x, params, y_pred);
            double mse = compute_mse(y, y_pred);
            std::cout << "Epoch " << epoch_idx << "/" << options.max_epochs
                      << ", MSE: " << mse << std::endl;
        }
    }
    return params;

}

}