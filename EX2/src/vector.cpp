#include "vector.hpp"

long int Scalar::operator()() const {
    long int sum = 0;
    for (int i = 0; i < sizeof(coefficients)/sizeof(coefficients[0]); ++i) {
        sum += coefficients[i];
    }
    return sum;
}

Scalar::Scalar(const Vector& vec) : size(vec.getSize()), coefficients(new int[vec.getSize()]) {
        assert(vec.getSize() == 10);
        for (int i = 0; i < vec.getSize(); ++i) {
            coefficients[i] = static_cast<int>(std::round(vec[i]));
        }
}