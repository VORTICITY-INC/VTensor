#pragma once

#include "lib/core/tensor.hpp"

namespace vt {

/**
 * @brief Expand the dimensions of the tensor on the left side.
 * 
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @tparam D: Number of dimensions to expand.
 * @param x: The tensor object.
 * @return Tensor<T, N + D> 
 */
template <typename T, size_t N, size_t D>
Tensor<T, N + D> expand_dims_lhs(const Tensor<T, N>& x) {
    if constexpr (D == 0) {
        return x;
    } else {
        return expand_dims_lhs<T, N + 1, D - 1>(x(vt::newaxis, vt::ellipsis));
    }
}

/**
 * @brief Expand the dimensions of the tensor on the right side.
 * 
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @tparam D: Number of dimensions to expand.
 * @param x: The tensor object.
 * @return Tensor<T, N + D> 
 */
template <typename T, size_t N, size_t D>
Tensor<T, N + D> expand_dims_rhs(const Tensor<T, N>& x) {
    if constexpr (D == 0) {
        return x;
    } else {
        return expand_dims_rhs<T, N + 1, D - 1>(x(vt::ellipsis, vt::newaxis));
    }
}

} // namespace vt
