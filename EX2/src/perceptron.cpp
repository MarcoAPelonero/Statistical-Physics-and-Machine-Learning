#include "perceptron.hpp"

#include <cassert>

Perceptron20::Perceptron20() {
    for (double& weight : J_) weight = 0;
}

Perceptron20::Perceptron20(const unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> dist(0.0, 1.0);
    for (double& weight : J_) weight = dist(rng);
}

Perceptron20 Perceptron20::perfectComparator10bits() {
    Perceptron20 p;
    for (int i = 0; i < 10; ++i) {
        const int value = 1 << (9 - i);
        p.J_[i] = value;           // weights for the top scalar
        p.J_[10 + i] = -value;     // mirror weights for the bottom scalar
    }
    return p;
}

int Perceptron20::eval(const Vector& S20) const {
    assert(S20.getSize() == 20);
    // Use floating-point accumulator so products between int inputs and
    // double weights are computed in double precision. The previous
    // implementation used a long long accumulator which caused the
    // double products to be truncated to integers, effectively losing
    // most weight information and biasing the sign output.
    double acc = 0.0;
    for (int i = 0; i < 20; ++i) {
        acc += static_cast<double>(S20[i]) * J_[i];
    }
    if (acc > 0.0) return +1;
    if (acc < 0.0) return -1;
    return 0;
}

int Perceptron20::compare(const Scalar& top, const Scalar& bottom) const {
    Vector S20 = Vector::concat(top.toVector(), bottom.toVector());
    return eval(S20);
}

std::array<double, 20> Perceptron20::weights() const {
    std::array<double, 20> out{};
    for (int i = 0; i < 20; ++i) out[i] = J_[i];
    return out;
}

void Perceptron20::applyUpdate(const Vector& S20, int label, double scale) {
    if (label == 0) return;
    const double delta = scale * static_cast<double>(label);
#ifdef _OPENMP
#pragma omp simd
#endif
    for (int i = 0; i < 20; ++i) {
        J_[i] += delta * static_cast<double>(S20[i]);
    }
}
