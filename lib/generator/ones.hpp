#pragma once

#include "lib/core/tensor.hpp"

namespace vt {

/**
 * @brief Returns a new array of given shape, filled with ones.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param shape: Shape of the tensor.
 * @param order: The order of the tensor.
 * @return Tensor: The new tensor object.
 */
template <typename T = float, size_t N>
Tensor<T, N> ones(Shape<N> shape, const Order order = Order::C) {
    using vector_type = typename Tensor<T, N>::vector_type;
    vector_type data(get_size(shape), T{1});
    return Tensor<T, N>(std::make_shared<vector_type>(data), shape, order);
}

/**
 * @brief Generate a 1D array of ones.
 *
 * @tparam T: Data type of the tensor.
 * @param m: Size of the tensor.
 * @param order: The order of the tensor.
 * @return Tensor<T, 1>: The new tensor object.
 */
template <typename T = float>
Tensor<T, 1> ones(size_t m, const Order order = Order::C) {
    return ones(Shape<1>{m}, order);
}

/**
 * @brief Generate a 2D array of ones.
 *
 * @tparam T: Data type of the tensor.
 * @param m: Number of rows of the tensor.
 * @param n: Number of columns of the tensor.
 * @param order: The order of the tensor.
 * @return Tensor<T, 2>: The new tensor object.
 */
template <typename T = float>
Tensor<T, 2> ones(size_t m, size_t n, const Order order = Order::C) {
    return ones<T, 2>(Shape<2>{m, n}, order);
}

}  // namespace vt
