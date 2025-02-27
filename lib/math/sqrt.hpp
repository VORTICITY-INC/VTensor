#pragma once

#include "lib/core/tensor.hpp"
#include "lib/generator/zeros.hpp"

namespace vt {

/**
 * @brief Returns the square root of the elements of the tensor.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param tensor: The tensor object.
 * @return Tensor<T, N>: The result tensor.
 */
template <typename T, size_t N>
Tensor<T, N> sqrt(const Tensor<T, N>& tensor) {
    auto result = vt::zeros<T>(tensor.shape(), tensor.order());
    thrust::transform(tensor.begin(), tensor.end(), result.begin(), [] __device__(const T& x) {
        if constexpr (std::is_same<T, float>::value)
            return sqrtf(x);
        else
            return sqrt(x);
    });
    return result;
}

}  // namespace vt
