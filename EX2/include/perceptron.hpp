#ifndef PERCEPTRON_HPP
#define PERCEPTRON_HPP

#include <array>

#include "vector.hpp"

class Perceptron20 {
public:
    Perceptron20();

    static Perceptron20 perfectComparator10bits();

    int eval(const Vector& S20) const;
    int compare(const Scalar& top, const Scalar& bottom) const;
    std::array<int, 20> weights() const;

private:
    int J_[20];
};

#endif // PERCEPTRON_HPP
