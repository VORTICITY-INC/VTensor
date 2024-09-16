#pragma once

#include <thrust/sequence.h>

#include <lib/core/tensor.hpp>
#include <lib/generator/zeros.hpp>

namespace vt {

/**
 * @brief Returns a new 1D array and filled with an increasing sequence.
 *
 * @tparam T: Data type of the tensor.
 * @param step: The size of the tensor.
 * @return Tensor: The new tensor object.
 */
template <typename T = float>
Tensor<T, 1> arange(int step) {
    assert(step > 0);
    auto tensor = zeros<T>(step);
    thrust::sequence(tensor.begin(), tensor.end());
    return tensor;
}

/**
 * @brief Generate a sequence of numbers from start to end with a given step size.
 *
 * @tparam T: Data type of the tensor.
 * @param start: The starting value of the sequence.
 * @param end: The end value of the sequence.
 * @param step: The step size of the sequence.
 * @return Tensor<T, 1>: The new tensor object.
 */
template <typename T = float>
Tensor<T, 1> arange(T start, T end, T step) {
    int n = static_cast<int>(std::ceil(double(end - start) / step));
    assert(n > 0);
    auto tensor = zeros<T>(n);
    thrust::transform(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(n), tensor.begin(),
                      [start, step] __device__(int i) -> T { return start + i * step; });
    return tensor;
}

/**
 * @brief Generate a sequence of pointers from start to end with a given step size.
 *
 * @tparam T: Data type of the tensor.
 * @param start: The starting value of the sequence.
 * @param end: The end value of the sequence.
 * @param step: The step size of the sequence.
 * @return Tensor<T*, 1>: The new tensor object.
 */
template <typename T>
Tensor<T*, 1> arange(T* start, T* end, int step) {
    int n = static_cast<int>(std::ceil(double(end - start) / step));
    assert(n > 0);
    auto tensor = zeros<T*>(n);
    thrust::transform(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(n), tensor.begin(),
                      [start, step] __device__(int i) -> T* { return start + i * step; });
    return tensor;
}

}  // namespace vt
