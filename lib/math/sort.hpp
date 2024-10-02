#pragma once

#include <thrust/sort.h>

#include <lib/core/assertions.hpp>
#include <lib/core/cutensor.hpp>
#include <lib/core/tensor.hpp>
#include <lib/math/transpose.hpp>
#include <lib/memory/copy.hpp>

namespace vt {

/**
 * @brief Sort the tensor.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param tensor: The tensor object.
 * @param copy: If True, the input tensor is copied before sorting.
 * @return Tensor<T, N>: The sorted tensor object.
 */
template <typename T, size_t N>
Tensor<T, N> sort(const Tensor<T, N>& tensor, bool copy = true) {
    auto result = copy ? vt::copy(tensor) : tensor;
    thrust::sort(result.begin(), result.end());
    return result;
}

/**
 * @brief Sort the tensor along the last axis.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param tensor: The tensor object.
 * @param axis: The axis to sort.
 * @return Tensor<T, N>: The sorted tensor object.
 */
template <typename T, size_t N>
void sort_along_the_last_axis(const Tensor<T, N>& tensor) {
    if constexpr (N == 1)
        sort(tensor, false);
    else
        for (auto i = 0; i < tensor.shape()[0]; i++) sort_along_the_last_axis(tensor[i]);
}

/**
 * @brief Sort the tensor along the given axis.
 * The method swaps the axis to the last dimension, and performs a for loop to sort along the last axis.
 * Current implementation is not optimized. The performance could improve if we sort the tensor along the given axis in parallel.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param tensor: The tensor object.
 * @param axis: The axis to sort.
 * @return Tensor<T, N>: The sorted tensor object.
 */
template <typename T, size_t N>
Tensor<T, N> sort(const Tensor<T, N>& tensor, int axis) {
    assert_at_least_1d_tensor<N>();
    auto result = moveaxis(copy(tensor), axis, N - 1);
    sort_along_the_last_axis(result);
    return moveaxis(result, N - 1, axis);
}

}  // namespace vt