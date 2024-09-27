#pragma once

#include <lib/core/cutensor.hpp>
#include <lib/core/tensor.hpp>
#include <lib/generator/zeros.hpp>

namespace vt {

/**
 * @brief Kernel to the tensor with ones at and below the given diagonal.
 *
 * @tparam T: Data type of the tensor.
 * @param tensor: The input tensor
 * @param k: An offset for the sub-diagonal at and below which the tensor is filled.
 */
template <typename T>
__global__ void tri_kernel(CuTensor<T, 2> tensor, int k) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < tensor.shape[0] && col < tensor.shape[1]) {
        if (row + k >= col) {
            tensor(row, col) = 1;
        }
    }
}

/**
 * @brief Return a tensor with ones at and below the given diagonal.
 *
 * @tparam T: Data type of the tensor.
 * @param m: Number of rows in the tensor.
 * @param n: Number of columns in the tensor.
 * @param k: An offset for the sub-diagonal at and below which the tensor is filled.
 * @return Tensor: The result tensor.
 */
template <typename T = float>
Tensor<T, 2> tri(size_t m, size_t n = 0, int k = 0) {
    if (n == 0) n = m;
    auto tensor = zeros<T>(m, n);
    dim3 tpb(16, 16);
    dim3 blk((m + tpb.x - 1) / tpb.x, (n + tpb.y - 1) / tpb.y);
    tri_kernel<T><<<blk, tpb>>>(tensor, k);
    return tensor;
}

/**
 * @brief Return a lower triangle of the tensor.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param tensor: The input tensor
 * @param k: An offset for the diagonal above which to zero elements.
 * @return Tensor: The result tensor.
 */
template <typename T, size_t N>
Tensor<T, N> tril(const Tensor<T, N>& tensor, int k = 0) {
    auto shape = tensor.shape();
    auto mask = tri<bool>(shape[N - 2], shape[N - 1], k);
    return where(mask, tensor, static_cast<T>(0));
}

/**
 * @brief Return an upper triangle of the tensor.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param tensor: The input tensor
 * @param k: An offset for the diagonal below which to zero elements.
 * @return Tensor: The result tensor.
 */
template <typename T, size_t N>
Tensor<T, N> triu(const Tensor<T, N>& tensor, int k = 0) {
    auto shape = tensor.shape();
    auto mask = tri<bool>(shape[N - 2], shape[N - 1], k - 1);
    return where(mask, static_cast<T>(0), tensor);
}

}  // namespace vt