#include "perceptron.hpp"

#include <cassert>

Perceptron20::Perceptron20() {
    for (int& weight : J_) weight = 0;
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
    long long acc = 0;
    for (int i = 0; i < 20; ++i) {
        acc += static_cast<long long>(S20[i]) * J_[i];
    }
    if (acc > 0) return +1;
    if (acc < 0) return -1;
    return 0;
}

int Perceptron20::compare(const Scalar& top, const Scalar& bottom) const {
    Vector S20 = Vector::concat(top.toVector(), bottom.toVector());
    return eval(S20);
}

std::array<int, 20> Perceptron20::weights() const {
    std::array<int, 20> out{};
    for (int i = 0; i < 20; ++i) out[i] = J_[i];
    return out;
}
