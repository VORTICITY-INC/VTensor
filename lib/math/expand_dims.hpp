#pragma once

#include "lib/core/tensor.hpp"

namespace vt {

template <typename T, size_t N, size_t D>
Tensor<T, N + D> expand_dims_lhs(const Tensor<T, N>& x) {
    if constexpr (D == 0) {
        return x;
    } else {
        return expand_dims_lhs<T, N + 1, D - 1>(x(vt::newaxis, vt::ellipsis));
    }
}

template <typename T, size_t N, size_t D>
Tensor<T, N + D> expand_dims_rhs(const Tensor<T, N>& x) {
    if constexpr (D == 0) {
        return x;
    } else {
        return expand_dims_rhs<T, N + 1, D - 1>(x(vt::ellipsis, vt::newaxis));
    }
}

} // namespace vt
