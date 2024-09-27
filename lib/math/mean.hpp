#pragma once

#include <lib/core/cutensor.hpp>
#include <lib/core/tensor.hpp>
#include <lib/math/reduce.hpp>
#include <lib/math/sum.hpp>

namespace vt {

/**
 * @brief Returns the mean value of the tensor elements.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param tensor: The tensor object.
 * @return T: The mean value of the tensor elements.
 */
template <typename T, size_t N>
double mean(const Tensor<T, N>& tensor) {
    return sum(tensor) / tensor.size();
}

/**
 * @brief Returns the mean value of how many elements are true in the tensor.
 *
 * @tparam N: Number of dimensions of the tensor.
 * @param tensor: The tensor object.
 * @return double: The mean value.
 */
template <size_t N>
double mean(const Tensor<bool, N>& tensor) {
    return double(sum(tensor)) / tensor.size();
}

/**
 * @brief Kernel to find the mean value of the tensor elements along the given axis.
 * The current implementation is not optimized. The performance could improve if a reduce algorithm is applied along the axis.
 *
 * @tparam T: The data type of the result tensor.
 * @tparam U: The data type of the input tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param tensor: The tensor object.
 * @param result: The tensor object to store the result.
 */
template <typename T, typename U, size_t N>
__global__ void mean_along_axis_kernel(CuTensor<U, N> tensor, CuTensor<T, N - 1> result) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= result.size) return;
    auto start = get_iterator_index<N - 1>(idx, tensor.shape, tensor.strides, tensor.start);
    auto sum = T{0};
    for (auto i = 0; i < tensor.shape[N - 1]; ++i) sum += tensor.data[start + i * tensor.strides[N - 1]];
    result[idx] = sum / tensor.shape[N - 1];
}

/**
 * @brief Find the mean value of the tensor elements along the given axis.
 * The current implementation is not optimized. The performance could improve if a reduce algorithm is applied along the axis.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param tensor: The tensor object.
 * @param axis: The axis to find the mean value.
 * @return Tensor<T, N-1>: The result tensor.
 */
template <typename T, size_t N>
Tensor<T, N - 1> mean(const Tensor<T, N>& tensor, const int axis) {
    return reduce_along_axis<T, T, N>(tensor, axis, mean_along_axis_kernel<T, T, N>);
}

/**
 * @brief Find the mean value of how many elements are true in the tensor along the given axis.
 * The current implementation is not optimized. The performance could improve if a reduce algorithm is applied along the axis.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param tensor: The tensor object.
 * @param axis: The axis to find the mean value.
 * @return Tensor<T, N-1>: The result tensor.
 */
template <typename T = float, size_t N>
Tensor<T, N - 1> mean(const Tensor<bool, N>& tensor, const int axis) {
    return reduce_along_axis<T, bool, N>(tensor, axis, mean_along_axis_kernel<T, bool, N>);
}

}  // namespace vt
