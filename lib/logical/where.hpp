#pragma once

#include "lib/core/tensor.hpp"

namespace vt {

/**
 * @brief Kernel to return elements chosen from x or y depending on condition.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @tparam M: Number of dimensions of the condition tensor.
 * @param cond: The condition tensor.
 * @param x: The tensor object to choose elements from when condition is true.
 * @param y: The tensor object to choose elements from when condition is false.
 * @param result: The result tensor.
 */
template <typename T, size_t N, size_t M>
__global__ void where_kernel(CuTensor<bool, M> cond, CuTensor<T, N> x, CuTensor<T, N> y, CuTensor<T, N> result) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= result.size) return;
    result[idx] = cond[idx % cond.size] ? x[idx] : y[idx];
}

/**
 * @brief Kernel to return elements chosen from x or y depending on condition.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @tparam M: Number of dimensions of the condition tensor.
 * @param cond: The condition tensor.
 * @param x: A constant value to choose when condition is true.
 * @param y: The tensor object to choose elements from when condition is false.
 * @param result: The result tensor.
 */
template <typename T, size_t N, size_t M>
__global__ void where_kernel(CuTensor<bool, M> cond, T x, CuTensor<T, N> y, CuTensor<T, N> result) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= result.size) return;
    result[idx] = cond[idx % cond.size] ? x : y[idx];
}

/**
 * @brief Kernel to return elements chosen from x or y depending on condition.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @tparam M: Number of dimensions of the condition tensor.
 * @param cond: The condition tensor.
 * @param x: The tensor object to choose elements from when condition is true.
 * @param y: A constant value to choose when condition is false.
 * @param result: The result tensor.
 */
template <typename T, size_t N, size_t M>
__global__ void where_kernel(CuTensor<bool, M> cond, CuTensor<T, N> x, T y, CuTensor<T, N> result) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= result.size) return;
    result[idx] = cond[idx % cond.size] ? x[idx] : y;
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
    auto shape = x.shape();
    auto cond_shape = cond.shape();
    for (size_t i = 0; i < M; i++) {
        assert(cond_shape[M - i - 1] == shape[N - i - 1]);
    }
    auto result = zeros<T>(shape);
    auto nblocks = (result.size() + NUM_THREADS_X - 1) / NUM_THREADS_X;
    where_kernel<T, N, M><<<nblocks, NUM_THREADS_X>>>(cond, x, y, result);
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
    auto cond_shape = cond.shape();
    for (size_t i = 0; i < M; i++) {
        assert(cond_shape[M - i - 1] == shape[N - i - 1]);
    }
    auto result = zeros<T>(shape);
    auto nblocks = (result.size() + NUM_THREADS_X - 1) / NUM_THREADS_X;
    where_kernel<T, N, M><<<nblocks, NUM_THREADS_X>>>(cond, x, y, result);
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
    auto cond_shape = cond.shape();
    for (size_t i = 0; i < M; i++) {
        assert(cond_shape[M - i - 1] == shape[N - i - 1]);
    }
    auto result = zeros<T>(shape);
    auto nblocks = (result.size() + NUM_THREADS_X - 1) / NUM_THREADS_X;
    where_kernel<T, N, M><<<nblocks, NUM_THREADS_X>>>(cond, x, y, result);
    return result;
}

}  // namespace vt