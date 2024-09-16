#pragma once

#include <lib/core/tensor.hpp>

namespace vt {

/**
 * @brief Operator to add two tensors and return new tensor.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param lhs: The left-hand side tensor.
 * @param rhs: The right-hand side tensor.
 * @return tensor: The new tensor object.
 */
template <typename T, size_t N>
Tensor<T, N> operator+(const Tensor<T, N>& lhs, const Tensor<T, N>& rhs) {
    assert(lhs.shape() == rhs.shape());
    auto result = Tensor<T, N>(lhs.shape());
    thrust::transform(lhs.begin(), lhs.end(), rhs.begin(), result.begin(), thrust::plus<T>());
    return result;
}

/**
 * @brief Operator to add a vector and a scalar.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param lhs: The left-hand side tensor.
 * @param value: The scalar value to be added.
 * @return tensor: The new tensor object.
 */
template <typename T, size_t N>
Tensor<T, N> operator+(const Tensor<T, N>& lhs, const T value) {
    auto result = Tensor<T, N>(lhs.shape());
    thrust::transform(lhs.begin(), lhs.end(), thrust::make_constant_iterator(value), result.begin(), thrust::plus<T>());
    return result;
}

/**
 * @brief Operator to add a vector and a scalar.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param value: The scalar value to be added.
 * @param rhs: The right-hand side tensor.
 * @return tensor: The new tensor object.
 */
template <typename T, size_t N>
Tensor<T, N> operator+(const T value, const Tensor<T, N>& rhs) {
    return rhs + value;
}

/**
 * @brief Operator to subtract two tensors and return new tensor.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param lhs: The left-hand side tensor.
 * @param rhs: The right-hand side tensor.
 * @return tensor: The new tensor object.
 */
template <typename T, size_t N>
Tensor<T, N> operator-(const Tensor<T, N>& lhs, const Tensor<T, N>& rhs) {
    assert(lhs.shape() == rhs.shape());
    auto result = Tensor<T, N>(lhs.shape());
    thrust::transform(lhs.begin(), lhs.end(), rhs.begin(), result.begin(), thrust::minus<T>());
    return result;
}

/**
 * @brief Operator to subtract a vector and a scalar.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param lhs: The left-hand side tensor.
 * @param value: The scalar value to be subtracted.
 * @return tensor: The new tensor object.
 */
template <typename T, size_t N>
Tensor<T, N> operator-(const Tensor<T, N>& lhs, const T value) {
    auto result = Tensor<T, N>(lhs.shape());
    thrust::transform(lhs.begin(), lhs.end(), thrust::make_constant_iterator(value), result.begin(), thrust::minus<T>());
    return result;
}

/**
 * @brief Operator to subtract a vector and a scalar.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param value: The scalar value.
 * @param rhs: The right-hand side tensor to be subtracted.
 * @return tensor: The new tensor object.
 */
template <typename T, size_t N>
Tensor<T, N> operator-(const T value, const Tensor<T, N>& rhs) {
    auto result = Tensor<T, N>(rhs.shape());
    auto iter = thrust::make_constant_iterator(value);
    thrust::transform(iter, iter + rhs.size(), rhs.begin(), result.begin(), thrust::minus<T>());
    return result;
}

/**
 * @brief Operator to multiply two tensors and return new tensor.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param lhs: The left-hand side tensor.
 * @param rhs: The right-hand side tensor.
 * @return tensor: The new tensor object.
 */
template <typename T, size_t N>
Tensor<T, N> operator*(const Tensor<T, N>& lhs, const Tensor<T, N>& rhs) {
    assert(lhs.shape() == rhs.shape());
    auto result = Tensor<T, N>(lhs.shape());
    thrust::transform(lhs.begin(), lhs.end(), rhs.begin(), result.begin(), thrust::multiplies<T>());
    return result;
}

/**
 * @brief Operator to multiply a tensor and a scalar.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param lhs: The left-hand side tensor.
 * @param value: The scalar value to be multiplied.
 * @return tensor: The new tensor object.
 */
template <typename T, size_t N>
Tensor<T, N> operator*(const Tensor<T, N>& lhs, const T value) {
    auto result = Tensor<T, N>(lhs.shape());
    thrust::transform(lhs.begin(), lhs.end(), thrust::make_constant_iterator(value), result.begin(), thrust::multiplies<T>());
    return result;
}

/**
 * @brief Operator to multiply a scalar and a tensor.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param value: The scalar value.
 * @param rhs: The right-hand side tensor to be multiplied.
 * @return tensor: The new tensor object.
 */
template <typename T, size_t N>
Tensor<T, N> operator*(const T value, const Tensor<T, N>& rhs) {
    return rhs * value;
}

/**
 * @brief Operator to divide two tensors and return new tensor.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param lhs: The left-hand side tensor.
 * @param rhs: The right-hand side tensor.
 * @return tensor: The new tensor object.
 */
template <typename T, size_t N>
Tensor<T, N> operator/(const Tensor<T, N>& lhs, const Tensor<T, N>& rhs) {
    assert(lhs.shape() == rhs.shape());
    auto result = Tensor<T, N>(lhs.shape());
    thrust::transform(lhs.begin(), lhs.end(), rhs.begin(), result.begin(), thrust::divides<T>());
    return result;
}

/**
 * @brief Operator to divide a tensor and a scalar.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param lhs: The left-hand side tensor.
 * @param value: The scalar value to be divided.
 * @return tensor: The new tensor object.
 */
template <typename T, size_t N>
Tensor<T, N> operator/(const Tensor<T, N>& lhs, const T value) {
    auto result = Tensor<T, N>(lhs.shape());
    thrust::transform(lhs.begin(), lhs.end(), thrust::make_constant_iterator(value), result.begin(), thrust::divides<T>());
    return result;
}

/**
 * @brief Operator to divide a scalar and a tensor.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param value: The scalar value.
 * @param rhs: The right-hand side tensor to be divided.
 * @return tensor: The new tensor object.
 */
template <typename T, size_t N>
Tensor<T, N> operator/(const T value, const Tensor<T, N>& rhs) {
    auto result = Tensor<T, N>(rhs.shape());
    auto iter = thrust::make_constant_iterator(value);
    thrust::transform(iter, iter + rhs.size(), rhs.begin(), result.begin(), thrust::divides<T>());
    return result;
}

/**
 * @brief Operator to perform greaten than comparison between two tensors.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param lhs: The left-hand side tensor.
 * @param value: The scalar value to be powered.
 * @return tensor: The new tensor object.
 */
template <typename T, size_t N>
Tensor<bool, N> operator>(const Tensor<T, N>& lhs, const Tensor<T, N>& rhs) {
    assert(lhs.shape() == rhs.shape());
    auto result = Tensor<bool, N>(lhs.shape());
    thrust::transform(lhs.begin(), lhs.end(), rhs.begin(), result.begin(), [] __device__(const int& x, const int& y) { return x > y; });
    return result;
}

/**
 * @brief Operator to perform greaten than comparison between a tensor and a scalar.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param lhs: The left-hand side tensor.
 * @param value: The scalar value to be compared.
 * @return tensor: The new tensor object.
 */
template <typename T, size_t N>
Tensor<bool, N> operator>(const Tensor<T, N>& lhs, const T value) {
    auto result = Tensor<bool, N>(lhs.shape());
    thrust::transform(lhs.begin(), lhs.end(), thrust::make_constant_iterator(value), result.begin(),
                      [] __device__(const int& x, const int& y) { return x > y; });
    return result;
}

/**
 * @brief Operator to perform greaten than comparison between a scalar and a tensor.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param value: The scalar value.
 * @param rhs: The right-hand side tensor.
 * @return tensor: The new tensor object.
 */
template <typename T, size_t N>
Tensor<bool, N> operator>(const T value, const Tensor<T, N>& rhs) {
    auto result = Tensor<bool, N>(rhs.shape());
    thrust::transform(rhs.begin(), rhs.end(), thrust::make_constant_iterator(value), result.begin(),
                      [] __device__(const int& x, const int& y) { return x < y; });
    return result;
}

/**
 * @brief Operator to perform less than comparison between two tensors.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param lhs: The left-hand side tensor.
 * @param rhs: The right-hand side tensor.
 * @return tensor: The new tensor object.
 */
template <typename T, size_t N>
Tensor<bool, N> operator<(const Tensor<T, N>& lhs, const Tensor<T, N>& rhs) {
    return rhs > lhs;
}

/**
 * @brief Operator to perform less than comparison between a tensor and a scalar.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param lhs: The left-hand side tensor.
 * @param value: The scalar value to be compared.
 * @return tensor: The new tensor object.
 */
template <typename T, size_t N>
Tensor<bool, N> operator<(const Tensor<T, N>& lhs, const T value) {
    return value > lhs;
}

/**
 * @brief Operator to perform less than comparison between a scalar and a tensor.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param value: The scalar value.
 * @param rhs: The right-hand side tensor.
 * @return tensor: The new tensor object.
 */
template <typename T, size_t N>
Tensor<bool, N> operator<(const T value, const Tensor<T, N>& rhs) {
    return rhs > value;
}

/**
 * @brief Operator to perform greaten than or equal comparison between two tensors.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param lhs: The left-hand side tensor.
 * @param rhs: The right-hand side tensor.
 * @return tensor: The new tensor object.
 */
template <typename T, size_t N>
Tensor<bool, N> operator>=(const Tensor<T, N>& lhs, const Tensor<T, N>& rhs) {
    assert(lhs.shape() == rhs.shape());
    auto result = Tensor<bool, N>(lhs.shape());
    thrust::transform(lhs.begin(), lhs.end(), rhs.begin(), result.begin(), [] __device__(const int& x, const int& y) { return x >= y; });
    return result;
}

/**
 * @brief Operator to perform greaten than or equal comparison between a tensor and a scalar.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param lhs: The left-hand side tensor.
 * @param value: The scalar value to be compared.
 * @return tensor: The new tensor object.
 */
template <typename T, size_t N>
Tensor<bool, N> operator>=(const Tensor<T, N>& lhs, const T value) {
    auto result = Tensor<bool, N>(lhs.shape());
    thrust::transform(lhs.begin(), lhs.end(), thrust::make_constant_iterator(value), result.begin(),
                      [] __device__(const int& x, const int& y) { return x >= y; });
    return result;
}

/**
 * @brief Operator to perform greaten than or equal comparison between a scalar and a tensor.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param value: The scalar value.
 * @param rhs: The right-hand side tensor.
 * @return tensor: The new tensor object.
 */
template <typename T, size_t N>
Tensor<bool, N> operator>=(const T value, const Tensor<T, N>& rhs) {
    auto result = Tensor<bool, N>(rhs.shape());
    thrust::transform(rhs.begin(), rhs.end(), thrust::make_constant_iterator(value), result.begin(),
                      [] __device__(const int& x, const int& y) { return x <= y; });
    return result;
}

/**
 * @brief Operator to perform less than or equal comparison between two tensors.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param lhs: The left-hand side tensor.
 * @param rhs: The right-hand side tensor.
 * @return tensor: The new tensor object.
 */
template <typename T, size_t N>
Tensor<bool, N> operator<=(const Tensor<T, N>& lhs, const Tensor<T, N>& rhs) {
    return rhs >= lhs;
}

/**
 * @brief Operator to perform less than or equal comparison between a tensor and a scalar.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param lhs: The left-hand side tensor.
 * @param value: The scalar value to be compared.
 * @return tensor: The new tensor object.
 */
template <typename T, size_t N>
Tensor<bool, N> operator<=(const Tensor<T, N>& lhs, const T value) {
    return value >= lhs;
}

/**
 * @brief Operator to perform less than or equal comparison between a scalar and a tensor.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param value: The scalar value.
 * @param rhs: The right-hand side tensor.
 * @return tensor: The new tensor object.
 */
template <typename T, size_t N>
Tensor<bool, N> operator<=(const T value, const Tensor<T, N>& rhs) {
    return rhs >= value;
}

/**
 * @brief Operator to perform element-wise equality comparison between two tensors.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param lhs: The left-hand side tensor.
 * @param rhs: The right-hand side tensor.
 * @return tensor: The new tensor object.
 */
template <typename T, size_t N>
Tensor<bool, N> operator==(const Tensor<T, N>& lhs, const Tensor<T, N>& rhs) {
    assert(lhs.shape() == rhs.shape());
    auto result = Tensor<bool, N>(lhs.shape());
    thrust::transform(lhs.begin(), lhs.end(), rhs.begin(), result.begin(), [] __device__(const int& x, const int& y) { return x == y; });
    return result;
}

/**
 * @brief Operator to perform element-wise equality comparison between a tensor and a scalar.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param lhs: The left-hand side tensor.
 * @param value: The scalar value to be compared.
 * @return tensor: The new tensor object.
 */
template <typename T, size_t N>
Tensor<bool, N> operator==(const Tensor<T, N>& lhs, const T value) {
    auto result = Tensor<bool, N>(lhs.shape());
    thrust::transform(lhs.begin(), lhs.end(), thrust::make_constant_iterator(value), result.begin(),
                      [] __device__(const int& x, const int& y) { return x == y; });
    return result;
}

/**
 * @brief Operator to perform element-wise equality comparison between a scalar and a tensor.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param value: The scalar value.
 * @param rhs: The right-hand side tensor.
 * @return tensor: The new tensor object.
 */
template <typename T, size_t N>
Tensor<bool, N> operator==(const T value, const Tensor<T, N>& rhs) {
    return rhs == value;
}

/**
 * @brief Operator to perform element-wise inequality comparison between two tensors.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param lhs: The left-hand side tensor.
 * @param rhs: The right-hand side tensor.
 * @return tensor: The new tensor object.
 */
template <typename T, size_t N>
Tensor<bool, N> operator!=(const Tensor<T, N>& lhs, const Tensor<T, N>& rhs) {
    assert(lhs.shape() == rhs.shape());
    auto result = Tensor<bool, N>(lhs.shape());
    thrust::transform(lhs.begin(), lhs.end(), rhs.begin(), result.begin(), [] __device__(const int& x, const int& y) { return x != y; });
    return result;
}

/**
 * @brief Operator to perform element-wise inequality comparison between a tensor and a scalar.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param lhs: The left-hand side tensor.
 * @param value: The scalar value to be compared.
 * @return tensor: The new tensor object.
 */
template <typename T, size_t N>
Tensor<bool, N> operator!=(const Tensor<T, N>& lhs, const T value) {
    auto result = Tensor<bool, N>(lhs.shape());
    thrust::transform(lhs.begin(), lhs.end(), thrust::make_constant_iterator(value), result.begin(),
                      [] __device__(const int& x, const int& y) { return x != y; });
    return result;
}

/**
 * @brief Operator to perform element-wise inequality comparison between a scalar and a tensor.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param value: The scalar value.
 * @param rhs: The right-hand side tensor.
 * @return tensor: The new tensor object.
 */
template <typename T, size_t N>
Tensor<bool, N> operator!=(const T value, const Tensor<T, N>& rhs) {
    return rhs != value;
}

}  // namespace vt
