#pragma once

#include <lib/core/tensor.hpp>
#include <lib/generator/zeros.hpp>
#include <vector>

namespace vt {

/**
 * @brief Copy the std::vector to the tensor.
 *
 * @tparam T: Data type of the tensor.
 * @param vector: The std::vector object.
 * @return Tensor: The tensor from the std::vector.
 */
template <typename T>
Tensor<T, 1> astensor(const std::vector<T>& vector) {
    auto tensor = zeros<T>(vector.size());
    thrust::copy(vector.begin(), vector.end(), tensor.begin());
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