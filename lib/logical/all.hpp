#pragma once

#include <thrust/count.h>
#include <thrust/logical.h>

#include <lib/core/cutensor.hpp>
#include <lib/core/tensor.hpp>
#include <lib/math/reduce.hpp>

namespace vt {

/**
 * @brief Check if all elements of the tensor are true.
 * thrust::all_of utilizes transform_reduce algorithm to check if all elements are true.
 * Looks like if the tensor is sliced, the iterator index will go beyond the tensor size, which will cause the function to return wrong answer.
 * The workaround is to use thrust::count_if to count the number of true elements.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param tensor: The tensor object.
 * @return bool: True if all elements are true, false otherwise.
 */
template <typename T, size_t N>
bool all(const Tensor<T, N>& tensor) {
    if (tensor.contiguous())
        return thrust::all_of(tensor.begin(), tensor.end(), thrust::identity<bool>());
    else
        return thrust::count_if(tensor.begin(), tensor.end(), thrust::identity<bool>()) == tensor.size();
}

/**
 * @brief Kernel to check if all elements of the tensor are true along the given axis.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param tensor: The tensor object.
 * @param result: The tensor object to store the result.
 */
template <typename T, size_t N>
__global__ void all_along_axis_kernel(vt::CuTensor<T, N> tensor, vt::CuTensor<bool, N - 1> result) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= result.size) return;
    auto start = get_iterator_index<N - 1>(idx, tensor.shape, tensor.strides, tensor.start);
    for (auto i = 0; i < tensor.shape[N - 1]; ++i) {
        if (!tensor.data[start + i * tensor.strides[N - 1]]) {
            result[idx] = false;
            return;
        }
    }
    result[idx] = true;
}

/**
 * @brief Check if all elements of the tensor are true along the given axis.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param tensor: The tensor object.
 * @param axis: The axis to check if all elements are true.
 * @return Tensor<bool, N-1>: The result tensor.
 */
template <typename T, size_t N>
Tensor<bool, N - 1> all(const Tensor<T, N>& tensor, const int axis) {
    return reduce_along_axis<bool, T, N>(tensor, axis, all_along_axis_kernel<T, N>);
}

}  // namespace vt