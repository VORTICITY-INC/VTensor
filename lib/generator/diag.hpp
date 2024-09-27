#pragma once

#include <lib/core/cutensor.hpp>
#include <lib/core/tensor.hpp>
#include <lib/generator/zeros.hpp>

namespace vt {

/**
 * @brief Kernel to generate a 2D tensor with its diagonal filled by a tensor.
 *
 * @tparam T: Data type of the tensor.
 * @param tensor: 1D diagnaoal tensor.
 * @param k: Index of diagonals.
 * @param result: The result tensor.
 */
template <typename T>
__global__ void gen_diag_kernel(CuTensor<T, 1> tensor, int k, CuTensor<T, 2> result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= tensor.size) return;
    if (k >= 0) {
        result(idx, idx + k) = tensor[idx];
    } else {
        result(idx - k, idx) = tensor[idx];
    }
}

/**
 * @brief Generate a 2D tensor with its diagonal filled by a tensor.
 *
 * @tparam T: Data type of the tensor.
 * @param tensor: 1D diagnaoal tensor.
 * @param k: Index of diagonals.
 * @return Tensor<T, 2>: The result tensor.
 */
template <typename T>
Tensor<T, 2> diag(Tensor<T, 1> tensor, int k = 0) {
    auto m = tensor.size() + std::abs(k);
    auto result = zeros<T>(m, m);
    auto nblocks = (m + NUM_THREADS_X - 1) / NUM_THREADS_X;
    gen_diag_kernel<T><<<nblocks, NUM_THREADS_X>>>(tensor, k, result);
    return result;
}

/**
 * @brief Kernel to get the diagonal of a 2D tensor.
 *
 * @tparam T: Data type of the tensor.
 * @param tensor: The input tensor.
 * @param k: Index of diagonals.
 * @param result: The result tensor.
 */
template <typename T>
__global__ void get_diag_kernel(CuTensor<T, 2> tensor, int k, CuTensor<T, 1> result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= result.size) return;
    if (k >= 0) {
        result[idx] = tensor(idx, idx + k);
    } else {
        result[idx] = tensor(idx - k, idx);
    }
}

/**
 * @brief Get the diagonal of a 2D tensor.
 *
 * @tparam T: Data type of the tensor.
 * @param tensor: The input tensor.
 * @param k: Index of diagonals.
 * @return Tensor<T, 1>: The result tensor.
 */
template <typename T>
Tensor<T, 1> diag(Tensor<T, 2> tensor, int k = 0) {
    auto [m, n] = tensor.shape();
    size_t ndiag = (k >= 0) ? std::min(m, n - k) : std::min(m + k, n);
    auto result = zeros<T>(ndiag);
    auto nblocks = (ndiag + NUM_THREADS_X - 1) / NUM_THREADS_X;
    get_diag_kernel<T><<<nblocks, NUM_THREADS_X>>>(tensor, k, result);
    return result;
}

}  // namespace vt