#pragma once

#include "lib/core/tensor.hpp"

namespace vt {

/// Forward declaration of the Tensor class.
template <typename T, size_t N>
class Tensor;

/**
 * @brief Convert the tensor to a new data type.
 *
 * @tparam T: Data type of the tensor.
 * @tparam U: The data type to cast.
 * @tparam N: Number of dimensions of the tensor.
 * @param tensor: The tensor object.
 * @return Tensor<U, N>: The new tensor object.
 */
template <typename T, typename U, size_t N>
Tensor<U, N> astype(Tensor<T, N> tensor) {
    if constexpr (std::is_same_v<T, U>) {
        return tensor;
    } else {
        auto result = Tensor<U, N>(tensor.shape(), tensor.order());
        thrust::transform(tensor.begin(), tensor.end(), result.begin(), [] __device__(const T& x) { return static_cast<U>(x); });
        return result;
    }
}

}  // namespace vt