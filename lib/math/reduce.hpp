#pragma once

#include "lib/core/assertions.hpp"
#include "lib/core/tensor.hpp"
#include "lib/generator/zeros.hpp"
#include "lib/math/transpose.hpp"

namespace vt {

/**
 * @brief Reduce the tensor along the given axis.
 * The method swaps the axis to the last dimension, and performs the reduction along the last axis.
 *
 * @tparam T: The data type of the result tensor.
 * @tparam U: The data type of the input tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @tparam KernelFunc: The kernel function to reduce the tensor.
 * @param tensor: The tensor object.
 * @param axis: The axis to reduce.
 * @param kernel_func: The kernel function to reduce the tensor.
 * @return Tensor<T, N-1>: The results tensor.
 */
template <typename T, typename U, size_t N, typename KernelFunc>
Tensor<T, N - 1> reduce_along_axis(const Tensor<U, N>& tensor, int axis, KernelFunc kernel_func) {
    assert_at_least_2d_tensor<N>();
    auto _tensor = moveaxis(tensor, axis, N - 1);
    Shape<N - 1> new_shape;
    auto shape = _tensor.shape();
    std::copy(shape.begin(), shape.end() - 1, new_shape.begin());
    auto result = zeros<T>(new_shape);
    auto nblocks = (result.size() + NUM_THREADS_X - 1) / NUM_THREADS_X;
    kernel_func<<<nblocks, NUM_THREADS_X>>>(_tensor, result);
    return result;
}

}  // namespace vt
