#ifndef POLY_FITTING_HPP
#define POLY_FITTING_HPP

#include <vector>
#include <functional>
#include <iostream>
#include "funcUtils.hpp"

void predict(
    const std::function<double(double, const std::vector<double>&)>& f,
    const std::vector<double>& x,
    const std::vector<double>& params,
    std::vector<double>& y_pred
);

namespace fit {

struct GDOptions {
    double lr = 0.01;
    std::size_t max_epochs = 1000;
    double l2 = 0.0;
    bool verbose = false;
    std::size_t print_every = 100;
};

struct SGDOptions: public GDOptions {
    std::size_t batch_size = 1;
    std::size_t decay_every = 100;
    double lr_decay = 1.0;
    bool drop_last = true;
};

void epoch(
    const std::function<double(double, const std::vector<double>&)>& f,
    const std::vector<double>& x,
    const std::vector<double>& y,
    std::vector<double>& params,
    double lr,
    double l2,
    double fd_eps = 1e-6
);

void stochastic_epoch(
    const std::function<double(double, const std::vector<double>&)>& f,
    const std::vector<double>& x,
    const std::vector<double>& y,
    std::vector<double>& params,
    double lr,
    double l2,
    rng::UniformIntRandom& rng,
    std::size_t batch_size,
    double lr_decay = 1.0,
    std::size_t decay_every = 100,
    bool drop_last = true,
    double fd_eps = 1e-6
);

std::vector<double> fit_gd(
    const std::function<double(double, const std::vector<double>&)>& f,
    const std::vector<double>& x,
    const std::vector<double>& y,
    std::vector<double> params0,
    const GDOptions& options,
    double fd_eps = 1e-6
);

std::vector<double> fit_sgd(
    const std::function<double(double, const std::vector<double>&)>& f,
    const std::vector<double>& x,
    const std::vector<double>& y,
    std::vector<double> params0,
    const SGDOptions& options,
    double fd_eps = 1e-6
);

}
#endif // POLY_FITTING_HPP