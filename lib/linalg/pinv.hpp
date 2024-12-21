#pragma once

#include "lib/core/broadcast.hpp"
#include "lib/core/tensor.hpp"
#include "lib/linalg/svd.hpp"
#include "lib/math/transpose.hpp"

namespace vt {

namespace linalg {

/**
 * @brief Compute the (Moore-Penrose) pseudo-inverse of a matrix.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param tensor: The tensor object to be inverted.
 * @param rcond: Cutoff parameter for small singular values.
 * @return Tensor<T, N>: The pseudo-inverse of the matrix.
 */
template <typename T, size_t N>
Tensor<T, N> pinv(Tensor<T, N>& tensor, T rcond = 1e-15) {
    assert(tensor.order() == vt::Order::C);
    auto [u, s, vt] = svd(tensor, false);
    auto cutoff = rcond * max(s, -1);

    auto leq = (s <= cutoff(ellipsis, newaxis));
    s = static_cast<T>(1) / s;
    s[leq] = static_cast<T>(0);

    auto _vt = swapaxes(vt, -2, -1);
    auto _s = s(ellipsis, newaxis) * swapaxes(u, -2, -1);

    return matmul(_vt, _s);
}

}  // namespace linalg

}  // namespace vt
