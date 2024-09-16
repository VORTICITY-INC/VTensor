#pragma once

#include <lib/core/tensor.hpp>
#include <lib/generator/zeros.hpp>

namespace vt {

/**
 * @brief Returns the power of the elements of the two tensors.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param lhs: The first tensor object.
 * @param rhs: The second tensor object.
 * @return Tensor<T, N>: The result tensor.
 */
template <typename T, size_t N>
Tensor<T, N> power(const Tensor<T, N>& lhs, const Tensor<T, N>& rhs) {
    assert(lhs.shape() == rhs.shape());
    auto result = vt::zeros<T>(lhs.shape());
    thrust::transform(lhs.begin(), lhs.end(), rhs.begin(), result.begin(), [] __device__(const T& x, const T& y) {
        if constexpr (std::is_same<T, float>::value)
            return powf(x, y);
        else
            return pow(x, y);
    });
    return result;
}

/**
 * @brief Returns the power of the elements of the tensor and a value.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param lhs: The tensor object.
 * @param value: The value.
 * @return Tensor<T, N>: The result tensor.
 */
template <typename T, size_t N>
Tensor<T, N> power(const Tensor<T, N>& lhs, const T value) {
    auto result = vt::zeros<T>(lhs.shape());
    thrust::transform(lhs.begin(), lhs.end(), thrust::make_constant_iterator(value), result.begin(), [] __device__(const T& x, const T& y) {
        if constexpr (std::is_same<T, float>::value)
            return powf(x, y);
        else
            return pow(x, y);
    });
    return result;
}

/**
 * @brief Returns the power of the value and the elements of the tensor.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param value: The value.
 * @param rhs: The tensor object.
 * @return Tensor<T, N>: The result tensor.
 */
template <typename T, size_t N>
Tensor<T, N> power(const T value, const Tensor<T, N>& rhs) {
    auto result = vt::zeros<T>(rhs.shape());
    thrust::transform(rhs.begin(), rhs.end(), thrust::make_constant_iterator(value), result.begin(), [] __device__(const T& x, const T& y) {
        if constexpr (std::is_same<T, float>::value)
            return powf(y, x);
        else
            return pow(y, x);
    });
    return result;
}

}  // namespace vt
