#ifndef VECTOR_HPP
#define VECTOR_HPP

#include <cassert>
#include <cmath>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <random>
#include <utility>

class Vector {
private:
    int size_;
    int* elements_;

public:
    // --- ctors/dtor ---
    explicit Vector(int s)
        : size_(s), elements_(new int[s]{}) {}

    Vector(const Vector& other)
        : size_(other.size_), elements_(new int[other.size_]) {
        for (int i = 0; i < size_; ++i) elements_[i] = other.elements_[i];
    }

    Vector(Vector&& other) noexcept
        : size_(other.size_), elements_(other.elements_) {
        other.size_ = 0;
        other.elements_ = nullptr;
    }

    Vector& operator=(Vector rhs) { // copy-and-swap
        swap(rhs);
        return *this;
    }

    ~Vector() { delete[] elements_; }

    // --- element access ---
    int& operator[](int idx) {
        assert(0 <= idx && idx < size_);
        return elements_[idx];
    }
    const int& operator[](int idx) const {
        assert(0 <= idx && idx < size_);
        return elements_[idx];
    }

    // --- utilities ---
    int getSize() const { return size_; }
    void fill(int v) {
        for (int i = 0; i < size_; ++i) elements_[i] = v;
    }
    int* data() { return elements_; }
    const int* data() const { return elements_; }
    void swap(Vector& other) noexcept {
        std::swap(size_, other.size_);
        std::swap(elements_, other.elements_);
    }

    // Build a 10-element +/-1 vector from an integer x in [0, 1023].
    // Most significant bit at index 0.
    static Vector fromInt10bits(int x) {
        assert(0 <= x && x < (1 << 10));
        Vector v(10);
        for (int i = 0; i < 10; ++i) {
            int bit = (x >> (9 - i)) & 1; // MSB first
            v[i] = bit ? +1 : -1;         // +/-1 encoding
        }
        return v;
    }

    // Concatenate two vectors
    static Vector concat(const Vector& a, const Vector& b) {
        Vector out(a.size_ + b.size_);
        for (int i = 0; i < a.size_; ++i) out[i] = a[i];
        for (int j = 0; j < b.size_; ++j) out[a.size_ + j] = b[j];
        return out;
    }
};

/**
 * Scalar: represents a 10-bit "binary number" stored as +/-1 coefficients.
 * operator() returns the integer value in [0, 1023].
 *
 * Mapping: Si in {-1,+1}  ->  bi = (Si + 1)/2 in {0,1}
 * value = sum_{i=0..9} bi * 2^{9-i}
 */
class Scalar {
private:
    int size_;
    int* coefficients_;

public:
    explicit Scalar(int s = 10)
        : size_(s), coefficients_(new int[s]{}) {
        assert(size_ == 10); // the exercise fixes 10 bits
    }

    // Build from Vector (expects size 10, elements in {-1,+1} or {0,1})
    explicit Scalar(const Vector& vec);

    Scalar(const Scalar& other)
        : size_(other.size_), coefficients_(new int[other.size_]) {
        for (int i = 0; i < size_; ++i) coefficients_[i] = other.coefficients_[i];
    }

    Scalar(Scalar&& other) noexcept
        : size_(other.size_), coefficients_(other.coefficients_) {
        other.size_ = 0;
        other.coefficients_ = nullptr;
    }

    Scalar& operator=(Scalar rhs) { // copy-and-swap
        swap(rhs);
        return *this;
    }

    ~Scalar() { delete[] coefficients_; }

    void swap(Scalar& other) noexcept {
        std::swap(size_, other.size_);
        std::swap(coefficients_, other.coefficients_);
    }

    // Return integer value 0..1023
    long int operator()() const;

    // Access
    int getSize() const { return size_; }
    const int& operator[](int idx) const {
        assert(0 <= idx && idx < size_);
        return coefficients_[idx];
    }
    int& operator[](int idx) {
        assert(0 <= idx && idx < size_);
        return coefficients_[idx];
    }
    const int* data() const { return coefficients_; }
    int* data() { return coefficients_; }

    // Helpers
    static Scalar fromInt(int x);       // make +/-1-encoded Scalar from 0..1023
    int toInt() const;                  // convert +/-1-encoded Scalar to 0..1023
    Vector toVector() const;            // export as Vector(10) +/-1
    void printBits(std::ostream& os) const {
        for (int i = 0; i < size_; ++i) os << (coefficients_[i] > 0 ? '1' : '0');
    }
};

#endif // VECTOR_HPP
