#ifndef VECTOR_HPP
#define VECTOR_HPP

#include <iostream>
#include <cmath>
#include <cassert>

class Scalar {
private:
    int size;
    int* coefficients;
public:
    Scalar(int s) : size(s), coefficients(new int[s]) {
        assert(s == 10);
    }
    ~Scalar() { delete[] coefficients; }
    
    long int operator()() const;
    // Build a scalar given a Vector object of coefficients
    Scalar(const Vector& vec);
};

class Vector {
private:
    int size;
    int* elements;
public:
    Vector(int s) : size(s), elements(new int[s]) {}
    ~Vector() { delete[] elements; }
    int& operator[](int index) { return elements[index]; }
    int getSize() const { return size; }
};

#endif // VECTOR_HPP