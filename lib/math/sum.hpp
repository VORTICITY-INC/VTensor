#pragma once

#include <thrust/reduce.h>

#include <lib/core/assertions.hpp>
#include <lib/core/cutensor.hpp>
#include <lib/core/tensor.hpp>
#include <lib/math/reduce.hpp>

namespace vt {

/**
 * @brief Returns the sum of the tensor elements.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param tensor: The tensor object.
 * @return Tensor<T, 0>: The result tensor.
 */
template <typename T, size_t N>
Tensor<T, 0> sum(const Tensor<T, N>& tensor) {
    auto result = Tensor<T, 0>(Shape<0>{});
    (*result.data())[0] = thrust::reduce(tensor.begin(), tensor.end(), (T)0, thrust::plus<T>());
    return result;
}

/**
 * @brief Returns the sum of how many elements are true in the tensor.
 *
 * @tparam N: Number of dimensions of the tensor.
 * @param tensor: The tensor object.
 * @return Tensor<T, 0>: The result tensor.
 */
template <size_t N>
Tensor<int, 0> sum(const Tensor<bool, N>& tensor) {
    auto result = Tensor<int, 0>(Shape<0>{});
    (*result.data())[0] = thrust::reduce(tensor.begin(), tensor.end(), 0, thrust::plus<int>());
    return result;
}

/**
 * @brief Kernel to sum the tensor elements along the given axis.
 * The current implementation is not optimized. The performance could improve if a reduce algorithm is applied along the axis.
 *
 * @tparam T: The data type of the result tensor.
 * @tparam U: The data type of the input tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param tensor: The tensor object.
 * @param result: The tensor object to store the result.
 */
template <typename T, typename U, size_t N>
__global__ void sum_along_axis_kernel(CuTensor<U, N> tensor, CuTensor<T, N - 1> result) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= result.size) return;
    auto start = get_iterator_index<N - 1>(idx, tensor.shape, tensor.strides, tensor.start);
    auto sum = T{0};
    for (auto i = 0; i < tensor.shape[N - 1]; ++i) sum += tensor.data[start + i * tensor.strides[N - 1]];
    result[idx] = sum;
}

/**
 * @brief Sum the tensor elements along the given axis.
 * The current implementation is not optimized. The performance could improve if a reduce algorithm is applied along the axis.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param tensor: The tensor object.
 * @param axis: The axis to sum.
 * @return Tensor<T, N-1>: The result tensor.
 */
template <typename T, size_t N>
Tensor<T, N - 1> sum(const Tensor<T, N>& tensor, int axis) {
    assert_at_least_1d_tensor<N>();
    if constexpr (N == 1) {
        return sum(tensor);
    } else {
        return reduce_along_axis<T, T, N>(tensor, axis, sum_along_axis_kernel<T, T, N>);
    }
}

/**
 * @brief Sum of how many elements are true along the given axis.
 * The current implementation is not optimized. The performance could improve if a reduce algorithm is applied along the axis.
 *
 * @tparam N: Number of dimensions of the tensor.
 * @param tensor: The tensor object.
 * @param axis: The axis to sum.
 * @return Tensor<int, N-1>: The result tensor.
 */
template <size_t N>
Tensor<int, N - 1> sum(const Tensor<bool, N>& tensor, int axis) {
    assert_at_least_1d_tensor<N>();
    if constexpr (N == 1) {
        return sum(tensor);
    } else {
        return reduce_along_axis<int, bool, N>(tensor, axis, sum_along_axis_kernel<int, bool, N>);
    }
}

}  // namespace vt
