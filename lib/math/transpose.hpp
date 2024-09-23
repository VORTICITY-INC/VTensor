#pragma once

#include <lib/core/tensor.hpp>

namespace vt {

/**
 * @brief Transpose the tensor.
 * After transpose, it is not quaranteed that the tensor is contiguous.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param tensor: The tensor object.
 * @return Tensor<T, N>: The transposed tensor.
 */
template <typename T, size_t N>
Tensor<T, N> transpose(const Tensor<T, N>& tensor) {
    Shape<N> new_shape;
    Shape<N> new_strides;
    auto shape = tensor.shape();
    auto strides = tensor.strides();
    for (size_t i = 0; i < N; i++) {
        new_shape[i] = shape[N - i - 1];
        new_strides[i] = strides[N - i - 1];
    }
    return Tensor(tensor.data(), new_shape, new_strides, tensor.start(), false);
}

/**
 * @brief Transpose the tensor with given axes.
 * After transpose, it is not quaranteed that the tensor is contiguous.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param tensor: The tensor object.
 * @param axes: The axes to transpose.
 * @return Tensor<T, N>: The transposed tensor.
 */
template <typename T, size_t N>
Tensor<T, N> transpose(const Tensor<T, N>& tensor, const Shape<N>& axes) {
    Shape<N> new_shape;
    Shape<N> new_strides;
    auto shape = tensor.shape();
    auto strides = tensor.strides();
    for (size_t i = 0; i < N; i++) {
        new_shape[i] = shape[axes[i]];
        new_strides[i] = strides[axes[i]];
    }
    return Tensor(tensor.data(), new_shape, new_strides, tensor.start(), false);
}

/**
 * @brief Swap the axis of the tensor.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param tensor: The tensor object.
 * @param source: The source axis.
 * @param destination: The destination axis.
 * @return Tensor<T, N>: The transposed tensor.
 */
template <typename T, size_t N>
Tensor<T, N> moveaxis(const Tensor<T, N>& tensor, const size_t source, const size_t destination) {
    assert(source < N && destination < N);
    Shape<N> axes;
    std::vector<size_t> _axes(N);
    for (auto i = 0; i < N; ++i) {
        _axes[i] = i;
    }
    auto temp = _axes[source];
    _axes.erase(_axes.begin() + source);
    _axes.insert(_axes.begin() + destination, temp);
    std::copy(_axes.begin(), _axes.end(), axes.begin());
    return transpose(tensor, axes);
}

}  // namespace vt
