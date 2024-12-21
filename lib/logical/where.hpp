#pragma once

#include "lib/core/tensor.hpp"
#include "lib/math/expand_dims.hpp"

namespace vt {

/**
 * @brief Kernel to return elements chosen from x or y depending on condition.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param cond: The condition tensor.
 * @param x: The tensor object to choose elements from when condition is true.
 * @param y: The tensor object to choose elements from when condition is false.
 * @param result: The result tensor.
 */
template <typename T, size_t N>
__global__ void where_kernel(CuTensor<bool, N> cond, CuTensor<T, N> x, CuTensor<T, N> y, CuTensor<T, N> result) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= result.size) return;
    result[idx] = cond[idx] ? x[idx] : y[idx];
}

/**
 * @brief Kernel to return elements chosen from x or y depending on condition.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param cond: The condition tensor.
 * @param x: A constant value to choose when condition is true.
 * @param y: The tensor object to choose elements from when condition is false.
 * @param result: The result tensor.
 */
template <typename T, size_t N>
__global__ void where_kernel(CuTensor<bool, N> cond, T x, CuTensor<T, N> y, CuTensor<T, N> result) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= result.size) return;
    result[idx] = cond[idx] ? x : y[idx];
}

/**
 * @brief Kernel to return elements chosen from x or y depending on condition.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param cond: The condition tensor.
 * @param x: The tensor object to choose elements from when condition is true.
 * @param y: A constant value to choose when condition is false.
 * @param result: The result tensor.
 */
template <typename T, size_t N>
__global__ void where_kernel(CuTensor<bool, N> cond, CuTensor<T, N> x, T y, CuTensor<T, N> result) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= result.size) return;
    result[idx] = cond[idx] ? x[idx] : y;
}

/**
 * @brief Return elements chosen from x or y depending on condition.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @tparam M: Number of dimensions of the condition tensor.
 * @param cond: The condition tensor.
 * @param x: The tensor object to choose elements from when condition is true.
 * @param y: The tensor object to choose elements from when condition is false.
 * @return Tensor: The result tensor.
 */
template <typename T, size_t N, size_t M>
Tensor<T, N> where(const Tensor<bool, M>& cond, const Tensor<T, N>& x, const Tensor<T, N>& y) {
    static_assert(N >= M);
    assert(x.shape() == y.shape());
    assert_same_order_between_two_tensors(x.order(), y.order());
    auto shape = x.shape();
    auto _cond = broadcast_to(expand_dims_lhs<bool, M, N - M>(cond), shape);
    auto result = zeros<T>(shape, x.order());
    auto nblocks = (result.size() + NUM_THREADS_X - 1) / NUM_THREADS_X;
    where_kernel<T, N><<<nblocks, NUM_THREADS_X>>>(_cond, x, y, result);
    return result;
}

/**
 * @brief Return elements chosen from x or y depending on condition.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @tparam M: Number of dimensions of the condition tensor.
 * @param cond: The condition tensor.
 * @param x: A constant value to choose when condition is true.
 * @param y: The tensor object to choose elements from when condition is false.
 * @return Tensor: The result tensor.
 */
template <typename T, size_t N, size_t M>
Tensor<T, N> where(const Tensor<bool, M>& cond, const T x, const Tensor<T, N>& y) {
    static_assert(N >= M);
    auto shape = y.shape();
    auto _cond = broadcast_to(expand_dims_lhs<bool, M, N - M>(cond), shape);
    auto result = zeros<T>(shape, y.order());
    auto nblocks = (result.size() + NUM_THREADS_X - 1) / NUM_THREADS_X;
    where_kernel<T, N><<<nblocks, NUM_THREADS_X>>>(_cond, x, y, result);
    return result;
}

/**
 * @brief Return elements chosen from x or y depending on condition.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @tparam M: Number of dimensions of the condition tensor.
 * @param cond: The condition tensor.
 * @param x: The tensor object to choose elements from when condition is true.
 * @param y: A constant value to choose when condition is false.
 * @return Tensor: The result tensor.
 */
template <typename T, size_t N, size_t M>
Tensor<T, N> where(const Tensor<bool, M>& cond, const Tensor<T, N>& x, const T y) {
    static_assert(N >= M);
    auto shape = x.shape();
    auto _cond = broadcast_to(expand_dims_lhs<bool, M, N - M>(cond), shape);
    auto result = zeros<T>(shape, x.order());
    auto nblocks = (result.size() + NUM_THREADS_X - 1) / NUM_THREADS_X;
    where_kernel<T, N><<<nblocks, NUM_THREADS_X>>>(_cond, x, y, result);
    return result;
}

}  // namespace vt