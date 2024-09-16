#pragma once

#include <lib/core/tensor.hpp>

namespace vt {

/**
 * @brief Copy the tensor from device to host's the std::vector.
 *
 * @param tensor: The tensor object.
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @return std::vector<T>: The std::vector from the tensor.
 */
template <typename T, size_t N>
std::vector<T> asvector(const Tensor<T, N>& tensor) {
    std::vector<T> vector(tensor.size());
    thrust::copy(tensor.begin(), tensor.end(), vector.begin());
    return vector;
}

}  // namespace vt
