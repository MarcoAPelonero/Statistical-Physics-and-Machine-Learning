#include "vector.hpp"

Scalar::Scalar(const Vector& vec)
    : size_(vec.getSize()), coefficients_(new int[vec.getSize()]) {
    assert(size_ == 10);
    for (int i = 0; i < size_; ++i) {
        int v = vec[i];
        if (v == 0) v = -1;
        if (v != -1 && v != +1) {
            v = (v >= 0) ? +1 : -1;
        }
        coefficients_[i] = v;
    }
}

long int Scalar::operator()() const {
    long int value = 0;
    for (int i = 0; i < size_; ++i) {
        int bi = (coefficients_[i] + 1) / 2;     // maps -1->0, +1->1
        value = (value << 1) | bi;
    }
    return value; // 0..1023
}

Scalar Scalar::fromInt(int x) {
    assert(0 <= x && x < (1 << 10));
    Scalar s(10);
    for (int i = 0; i < 10; ++i) {
        int bit = (x >> (9 - i)) & 1;      // MSB first
        s[i] = bit ? +1 : -1;              // ±1 encoding
    }
    return s;
}

int Scalar::toInt() const {
    int value = 0;
    for (int i = 0; i < size_; ++i) {
        int bi = (coefficients_[i] + 1) / 2;     // maps -1->0, +1->1
        value = (value << 1) | bi;
    }
    return value; // 0..1023
}

Vector Scalar::toVector() const {
    Vector v(size_);
    for (int i = 0; i < size_; ++i) v[i] = coefficients_[i];
    return v;
}
