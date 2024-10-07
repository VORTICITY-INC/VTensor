#pragma once

#include "lib/core/tensor.hpp"
#include "lib/generator/zeros.hpp"

namespace vt {

/**
 * @brief Copy the tensor.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param tensor: The tensor object.
 * @return Tensor: The copied tensor object.
 */
template <typename T, size_t N>
Tensor<T, N> copy(const Tensor<T, N>& tensor) {
    auto tensor_cp = zeros<T>(tensor.shape());
    thrust::copy(tensor.begin(), tensor.end(), tensor_cp.begin());
    return tensor_cp;
}

}  // namespace vt