#pragma once

#include "lib/core/cutensor.hpp"
#include "lib/core/tensor.hpp"
#include "lib/generator/zeros.hpp"

namespace vt {

/**
 * @brief Kernel to set the diagonal elements to 1.
 *
 * @tparam T: Data type of the tensor.
 * @param tensor: Tensor object
 * @param ndiag: Number of diagonals needed to be set to 1.
 * @return __global__
 */
template <typename T>
__global__ void eye_kernel(CuTensor<T, 2> tensor, size_t ndiag) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ndiag) return;
    tensor(idx, idx) = 1;
}

/**
 * @brief Generate a 2D array with ones on the diagonals and zeros elsewhere.
 *
 * @tparam T: Data type of the tensor.
 * @param m: The number of rows.
 * @param n: The number of columns.
 * @param order: The order of the tensor.
 * @return Tensor<T, 2>: The new tensor object.
 */
template <typename T = float>
Tensor<T, 2> eye(size_t m, size_t n, const Order order = Order::C) {
    auto tensor = zeros<T>(m, n, order);
    size_t ndiag = std::min(m, n);
    auto nblocks = (ndiag + NUM_THREADS_X - 1) / NUM_THREADS_X;
    eye_kernel<T><<<nblocks, NUM_THREADS_X>>>(tensor, ndiag);
    return tensor;
}

/**
 * @brief Generate a 2D identity matrix
 *
 * @tparam T: Data type of the tensor.
 * @param m: The number of rows and columns.
 * @param order: The order of the tensor.
 * @return Tensor<T, 2>: The new tensor object.
 */
template <typename T = float>
Tensor<T, 2> eye(size_t m, const Order order = Order::C) {
    return eye<T>(m, m, order);
}

}  // namespace vt
