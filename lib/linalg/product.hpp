#pragma once

#include <thrust/inner_product.h>

#include <lib/core/tensor.hpp>

namespace vt {

/**
 * @brief Returns the dot product of two tensors.
 *
 * @tparam T: Data type of the tensors.
 * @tparam N: Number of dimensions of the tensor.
 * @param tensor1: The first tensor.
 * @param tensor2: The second tensor.
 * @return T: The dot product of the two tensors.
 */
template <typename T, size_t N>
T dot(Tensor<T, N> tensor1, Tensor<T, N> tensor2) {
    assert(tensor1.shape() == tensor2.shape());
    auto re = thrust::inner_product(tensor1.begin(), tensor1.end(), tensor2.begin(), T{0.0});
    return re;
}

}  // namespace vt
