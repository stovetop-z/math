#ifndef TENSOR_H
#define TENSOR_H

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <algorithm>

template<typename T, size_t N>
class Tensor {
private:
    T* tensor;
    size_t shape[N];
    size_t stride[N];
    size_t length;

    void computeStride();
    size_t offSet(const size_t (&location)[N]) const;

public:
    Tensor();
    Tensor(const size_t (&dimensions)[N]);
    Tensor(const Tensor& t);
    ~Tensor();

    // Getters for encapsulation
    const size_t* getShape() const { return shape; }
    size_t getLength() const { return length; }

    // Element access (non-const and const)
    T& byIndex(size_t i);
    const T& byIndex(size_t i) const;

    T& at(const size_t (&location)[N]);
    const T& at(const size_t (&location)[N]) const;

    T& operator()(const size_t (&location)[N]);
    const T& operator()(const size_t (&location)[N]) const;

    T& operator[](size_t i);
    const T& operator[](size_t i) const;

    // Assignment operators
    Tensor& operator=(const Tensor& t);
    Tensor& operator=(const T* t);

    template<size_t S>
    Tensor& operator=(const T (&t)[S]);
};

#include "tensor.cc"
#endif