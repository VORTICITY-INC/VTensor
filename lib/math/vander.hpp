#pragma once

#include "lib/core/tensor.hpp"
#include "lib/math/expand_dims.hpp"

namespace vt {

/**
 * @brief Compute the Vandermonde matrix of a tensor.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param x: Tensor object.
 * @param degree: The degree of the Vandermonde matrix.
 * @return Tensor<T, 0>: The result tensor.
 */
template <typename T, size_t N>
Tensor<T, N + 1> vander(const Tensor<T, N>& x, const int degree) {
    auto seq = arange(static_cast<T>(degree), T{-1.0}, T{-1.0});
    auto _seq = expand_dims_lhs<T, 1, N>(seq);
    auto _x = expand_dims_rhs<T, N, 1>(x);
    auto re = power(_x, _seq);
    return re;
}

}  // namespace vt