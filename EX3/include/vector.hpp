#ifndef VECTOR_HPP
#define VECTOR_HPP

#include <cassert>
#include <cmath>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <random>
#include <utility>

// Generic vector container with a default element type of int so existing
// code using `Vector` without template arguments keeps working.
template <typename T = int>
class Vector {
private:
    int size_;
    T* elements_;

public:
    // --- ctors/dtor ---
    explicit Vector(int s)
        : size_(s), elements_(new T[s]{}) {}

    Vector(const Vector& other)
        : size_(other.size_), elements_(new T[other.size_]) {
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
    T& operator[](int idx) {
        assert(0 <= idx && idx < size_);
        return elements_[idx];
    }
    const T& operator[](int idx) const {
        assert(0 <= idx && idx < size_);
        return elements_[idx];
    }

    // --- utilities ---
    int getSize() const { return size_; }
    void fill(T v) {
        for (int i = 0; i < size_; ++i) elements_[i] = v;
    }
    T* data() { return elements_; }
    const T* data() const { return elements_; }
    void swap(Vector& other) noexcept {
        std::swap(size_, other.size_);
        std::swap(elements_, other.elements_);
    }

    // Build a +/-1 vector from an integer x in [0, 2^bits - 1].
    // Most significant bit at index 0. This factory returns a Vector<int>.
    static Vector<int> fromIntBits(int x, int bits) {
        assert(bits > 0);
        assert(0 <= x && x < (1 << bits));
        Vector<int> v(bits);
        for (int i = 0; i < bits; ++i) {
            int bit = (x >> (bits - 1 - i)) & 1; // MSB first
            v[i] = bit ? +1 : -1;                // +/-1 encoding
        }
        return v;
    }

    // Concatenate two vectors (same element type)
    static Vector concat(const Vector& a, const Vector& b) {
        Vector out(a.size_ + b.size_);
        for (int i = 0; i < a.size_; ++i) out[i] = a[i];
        for (int j = 0; j < b.size_; ++j) out[a.size_ + j] = b[j];
        return out;
    }
};

/**
 * Scalar: represents a variable-length "binary number" stored as +/-1
 * coefficients. The integer conversion methods use the current scalar size
 * so values range from 0 .. (1<<size)-1.
 *
 * Mapping: Si in {-1,+1}  ->  bi = (Si + 1)/2 in {0,1}
 */
class Scalar {
private:
    int size_;
    int* coefficients_;

public:
    explicit Scalar(int s = 10)
        : size_(s), coefficients_(new int[s]{}) {
        // allow variable bit-length scalars (default kept for compatibility)
    }

    // Build from Vector<int> (expects any size, elements in {-1,+1} or {0,1})
    explicit Scalar(const Vector<int>& vec);

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

    // Return integer value 0 .. (1<<size_)-1
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
    // make +/-1-encoded Scalar from 0 .. (1<<bits)-1; default bits=10
    static Scalar fromInt(int x, int bits = 10);
    int toInt() const;                  // convert +/-1-encoded Scalar to integer
    Vector<int> toVector() const;       // export as Vector(size_) +/-1
    void printBits(std::ostream& os) const {
        for (int i = 0; i < size_; ++i) os << (coefficients_[i] > 0 ? '1' : '0');
    }
};

#endif // VECTOR_HPP
