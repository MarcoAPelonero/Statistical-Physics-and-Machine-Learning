#ifndef PERCEPTRON_HPP
#define PERCEPTRON_HPP

#include <array>
#include <random>
#include "vector.hpp"

class Perceptron20 {
public:
    Perceptron20();
    Perceptron20(const unsigned seed);

    static Perceptron20 perfectComparator10bits();

    int eval(const Vector& S20) const;
    int compare(const Scalar& top, const Scalar& bottom) const;
    std::array<double, 20> weights() const;
    void applyUpdate(const Vector& S20, int label, double scale);

private:
    double J_[20];
};

#endif // PERCEPTRON_HPP
