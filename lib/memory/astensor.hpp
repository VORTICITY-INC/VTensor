#pragma once

#include <vector>

#include "lib/core/tensor.hpp"
#include "lib/generator/zeros.hpp"
#include "xtensor/xarray.hpp"

namespace vt {

/**
 * @brief Copy std::vector to the tensor.
 *
 * @tparam T: Data type of the tensor.
 * @param vector: std::vector object.
 * @return Tensor: The tensor from std::vector.
 */
template <typename T>
Tensor<T, 1> astensor(const std::vector<T>& vector) {
    auto tensor = zeros<T>(vector.size());
    thrust::copy(vector.begin(), vector.end(), tensor.begin());
    return tensor;
}

/**
 * @brief Copy xarray to the tensor.
 *
 * @tparam T: Data type of the tensor.
 * @param arr: The xarray object.
 * @return Tensor: The tensor from the xarray.
 */
template <typename T, size_t N>
Tensor<T, N> astensor(const xt::xarray<T>& arr) {
    auto s = arr.shape();
    vt::Shape<N> shape;
    assert(s.size() == N);
    std::copy(s.begin(), s.end(), shape.begin());
    auto tensor = zeros<T>(shape);
    thrust::copy(arr.begin(), arr.end(), tensor.begin());
    return tensor;
}

/**
 * @brief Copy the raw pointer to the tensor.
 *
 * @tparam T: Data type of the tensor.
 * @param ptr: The raw pointer.
 * @param size: The size of the tensor.
 * @return Tensor: The tensor from the raw pointer.
 */
template <typename T>
Tensor<T, 1> astensor(const T* ptr, const size_t size) {
    auto tensor = zeros<T>(size);
    thrust::copy(ptr, ptr + size, tensor.begin());
    return tensor;
}

}  // namespace vt