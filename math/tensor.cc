#include "tensor.h"

template<typename T, size_t N>
Tensor<T, N>::Tensor() : tensor(nullptr), length(0) {
    std::memset(shape, 0, sizeof(shape));
    std::memset(stride, 0, sizeof(stride));
}

template<typename T, size_t N>
Tensor<T, N>::Tensor(const size_t (&dimensions)[N]) : tensor(nullptr), length(1) {
    for (size_t i = 0; i < N; i++) {
        shape[i] = dimensions[i];
        length *= dimensions[i];
    }
    computeStride();
    tensor = new T[length]();
}

template<typename T, size_t N>
Tensor<T, N>::Tensor(const Tensor& t) : length(t.length) {
    std::memcpy(shape, t.shape, sizeof(shape));
    std::memcpy(stride, t.stride, sizeof(stride));
    tensor = new T[length];
    std::copy(t.tensor, t.tensor + length, tensor);
}

template<typename T, size_t N>
Tensor<T, N>::~Tensor() {
    delete[] tensor;
}

template<typename T, size_t N>
void Tensor<T, N>::computeStride() {
    if (N == 0) return;
    stride[N - 1] = 1;
    for (int i = static_cast<int>(N) - 2; i >= 0; i--) {
        stride[i] = stride[i + 1] * shape[i + 1];
    }
}

template<typename T, size_t N>
size_t Tensor<T, N>::offSet(const size_t (&location)[N]) const {
    size_t offset = 0;
    for (size_t i = 0; i < N; i++) {
        offset += location[i] * stride[i];
    }
    return offset;
}

template<typename T, size_t N>
T& Tensor<T, N>::byIndex(size_t i) { return tensor[i]; }

template<typename T, size_t N>
const T& Tensor<T, N>::byIndex(size_t i) const { return tensor[i]; }

template<typename T, size_t N>
T& Tensor<T, N>::at(const size_t (&location)[N]) { return tensor[offSet(location)]; }

template<typename T, size_t N>
const T& Tensor<T, N>::at(const size_t (&location)[N]) const { return tensor[offSet(location)]; }

template<typename T, size_t N>
T& Tensor<T, N>::operator()(const size_t (&location)[N]) { return tensor[offSet(location)]; }

template<typename T, size_t N>
const T& Tensor<T, N>::operator()(const size_t (&location)[N]) const { return tensor[offSet(location)]; }

template<typename T, size_t N>
T& Tensor<T, N>::operator[](size_t i) { return tensor[i]; }

template<typename T, size_t N>
const T& Tensor<T, N>::operator[](size_t i) const { return tensor[i]; }

template<typename T, size_t N>
Tensor<T, N>& Tensor<T, N>::operator=(const T* t) {
    if (t && tensor) {
        for (size_t i = 0; i < length; i++) {
            tensor[i] = t[i];
        }
    }
    return *this;
}

template<typename T, size_t N>
template<size_t S>
Tensor<T, N>& Tensor<T, N>::operator=(const T (&t)[S]) {
    if (S == length && tensor) {
        for (size_t i = 0; i < length; i++) {
            tensor[i] = t[i];
        }
    }
    return *this;
}

template<typename T, size_t N>
Tensor<T, N>& Tensor<T, N>::operator=(const Tensor& t) {
    if (this == &t) return *this;

    if (length != t.length) {
        delete[] tensor;
        length = t.length;
        tensor = new T[length];
    }
    std::memcpy(shape, t.shape, sizeof(shape));
    std::memcpy(stride, t.stride, sizeof(stride));
    std::copy(t.tensor, t.tensor + length, tensor);
    return *this;
}