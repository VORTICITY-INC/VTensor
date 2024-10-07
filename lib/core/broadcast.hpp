#pragma once

#include <tuple>

#include "lib/core/assertions.hpp"
#include "lib/core/tensor.hpp"

namespace vt {

/**
 * @brief Broadcast two tensors to a common shape.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param tensor1: The first tensor.
 * @param tensor2: The second tensor.
 * @return std::tuple<Tensor<T, N>, Tensor<T, N>>: The broadcasted tensors.
 */
template <typename T, size_t N>
std::tuple<Tensor<T, N>, Tensor<T, N>> broadcast(const Tensor<T, N>& tensor1, const Tensor<T, N>& tensor2) {
    if constexpr (N > 0) {
        auto shape1 = tensor1.shape();
        auto shape2 = tensor2.shape();
        if (shape1 == shape2) {
            return std::make_tuple(tensor1, tensor2);
        }
        for (size_t i = 0; i < N; ++i) {
            assert(shape1[i] == shape2[i] || shape1[i] == 1 || shape2[i] == 1);
        }
        Shape<N> shape;
        for (size_t i = 0; i < N; ++i) {
            shape[i] = std::max(shape1[i], shape2[i]);
        }
        auto t1 = broadcast_to(tensor1, shape);
        auto t2 = broadcast_to(tensor2, shape);
        return std::make_tuple(t1, t2);
    } else {
        return std::make_tuple(tensor1, tensor2);
    }
}

/**
 * @brief Broadcast a tensor to a new shape.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param tensor: The tensor to be broadcasted.
 * @param broadcast_shape: The new shape of the tensor.
 * @return tensor: The new tensor object.
 */
template <typename T, size_t N>
Tensor<T, N> broadcast_to(const Tensor<T, N>& tensor, const Shape<N>& broadcast_shape) {
    assert_at_least_1d_tensor<N>();
    auto shape = tensor.shape();
    auto strides = tensor.strides();
    for (size_t i = 0; i < N; ++i) {
        assert(shape[i] == broadcast_shape[i] || shape[i] == 1);
    }
    Shape<N> broadcast_strides;
    for (size_t i = 0; i < N; ++i) {
        if (tensor.shape()[i] == 1 && broadcast_shape[i] > 1) {
            broadcast_strides[i] = 0;
        } else {
            broadcast_strides[i] = strides[i];
        }
    }
    return Tensor<T, N>(tensor.data(), broadcast_shape, broadcast_strides, tensor.start(), false);
}

}  // namespace vt