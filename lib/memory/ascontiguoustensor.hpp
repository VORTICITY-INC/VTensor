#pragma once

#include "lib/core/tensor.hpp"
#include "lib/memory/copy.hpp"

namespace vt {

/**
 * @brief Convert the tensor to a contiguous tensor.
 * If the tensor is not contiguous, it will copy and return a new tensor. Otherwise, it will return the same tensor.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param tensor: The tensor object.
 * @return Tensor: The contiguous tensor object.
 */
template <typename T, size_t N>
Tensor<T, N> ascontiguoustensor(const Tensor<T, N>& tensor) {
    if (!tensor.contiguous()) {
        return copy(tensor);
    } else {
        return tensor;
    }
}
}  // namespace vt