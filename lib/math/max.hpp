#pragma once

#include <thrust/extrema.h>

#include "lib/core/assertions.hpp"
#include "lib/core/cutensor.hpp"
#include "lib/core/tensor.hpp"
#include "lib/math/reduce.hpp"

namespace vt {

/**
 * @brief Returns the maximum value of the tensor elements.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param tensor: The tensor object.
 * @return Tensor<T, 0>: The result tensor.
 */
template <typename T, size_t N>
Tensor<T, 0> max(const Tensor<T, N>& tensor) {
    auto result = Tensor<T, 0>(Shape<0>{});
    (*result.data())[0] = *thrust::max_element(tensor.begin(), tensor.end());
    return result;
}

/**
 * @brief Kernel to find the maximum value of the tensor elements along the given axis.
 * The current implementation is not optimized. The performance could improve if a scan algorithm is applied along the axis.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param tensor: The tensor object.
 * @param result: The tensor object to store the result.
 */
template <typename T, size_t N>
__global__ void max_along_axis_kernel(CuTensor<T, N> tensor, CuTensor<T, N - 1> result) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= result.size) return;
    auto start = get_iterator_index<N - 1>(idx, tensor.shape, tensor.strides, tensor.start);
    auto value = tensor.data[start];
    for (auto i = 0; i < tensor.shape[N - 1]; ++i) value = std::max(value, tensor.data[start + i * tensor.strides[N - 1]]);
    result[idx] = value;
}

/**
 * @brief Find the maximum value of the tensor elements along the given axis.
 * The current implementation is not optimized. The performance could improve if a scan algorithm is applied along the axis.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param tensor: The tensor object.
 * @param axis: The axis to find the maximum value.
 * @return Tensor<T, N-1>: The result tensor.
 */
template <typename T, size_t N>
Tensor<T, N - 1> max(const Tensor<T, N>& tensor, int axis) {
    assert_at_least_1d_tensor<N>();
    if constexpr (N == 1) {
        return max(tensor);
    } else {
        return reduce_along_axis<T, T, N>(tensor, axis, max_along_axis_kernel<T, N>);
    }
}

}  // namespace vt
