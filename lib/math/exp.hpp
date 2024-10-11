#pragma once

#include "lib/core/tensor.hpp"
#include "lib/generator/zeros.hpp"

namespace vt {

/**
 * @brief Returns the exponential of the elements of the tensor.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param tensor: The tensor object.
 * @return Tensor<T, N>: The result tensor.
 */
template <typename T, size_t N>
Tensor<T, N> exp(const Tensor<T, N>& tensor) {
    auto result = vt::zeros<T>(tensor.shape());
    thrust::transform(tensor.begin(), tensor.end(), result.begin(), [] __device__(const T& x) {
        if constexpr (std::is_same<T, float>::value)
            return expf(x);
        else
            return exp(x);
    });
    return result;
}

}  // namespace vt
