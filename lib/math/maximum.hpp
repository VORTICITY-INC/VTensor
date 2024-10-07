#pragma once

#include <thrust/transform.h>

#include "lib/core/tensor.hpp"
#include "lib/generator/zeros.hpp"

namespace vt {

/**
 * @brief Element-wise maximum of two tensors.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param lhs: The first tensor object.
 * @param rhs: The second tensor object.
 * @return Tensor<T, N>: The result tensor.
 */
template <typename T, size_t N>
Tensor<T, N> maximum(const Tensor<T, N>& lhs, const Tensor<T, N>& rhs) {
    assert(lhs.shape() == rhs.shape());
    auto result = zeros<T>(lhs.shape());
    thrust::transform(lhs.begin(), lhs.end(), rhs.begin(), result.begin(), thrust::maximum<T>());
    return result;
}

/**
 * @brief Element-wise maximum of a tensor and a scalar value.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param lhs: The tensor object.
 * @param value: The scalar value.
 * @return Tensor<T, N>: The result tensor.
 */
template <typename T, size_t N>
Tensor<T, N> maximum(const Tensor<T, N>& lhs, const T value) {
    auto result = zeros<T>(lhs.shape());
    thrust::transform(lhs.begin(), lhs.end(), thrust::make_constant_iterator(value), result.begin(), thrust::maximum<T>());
    return result;
}

/**
 * @brief Element-wise maximum of a scalar value and a tensor.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param value: The scalar value.
 * @param rhs: The tensor object.
 * @return Tensor<T, N>: The result tensor.
 */
template <typename T, size_t N>
Tensor<T, N> maximum(const T value, const Tensor<T, N>& rhs) {
    return maximum(rhs, value);
}

}  // namespace vt
