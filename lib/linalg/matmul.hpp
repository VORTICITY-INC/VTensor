#pragma once

#include "lib/core/assertions.hpp"
#include "lib/core/tensor.hpp"
#include "lib/generator/zeros.hpp"
#include "lib/linalg/cublas.hpp"
#include "lib/memory/ascontiguoustensor.hpp"

namespace vt {

/**
 * @brief Performs a dot product between two 1D tensors using CuBLAS functions.
 *
 * @tparam T: Data type of the tensors.
 * @param tensor1: The first tensor.
 * @param tensor2: The second tensor.
 * @param handle: The CuBLAS handle. The default is the global CuBLAS handle.
 * @return Tensor: The result tensor.
 */
template <typename T>
Tensor<T, 1> matmul(Tensor<T, 1>& tensor1, Tensor<T, 1>& tensor2, cublasHandle_t handle = cuda::cublas.get_handle()) {
    assert(tensor1.order() == vt::Order::C);
    assert(tensor2.order() == vt::Order::C);
    assert(tensor1.size() == tensor2.size());
    auto t1 = ascontiguoustensor(tensor1);
    auto t2 = ascontiguoustensor(tensor2);
    auto dot = cuda::CuBLASFunc<T>::dot();
    Tensor<T, 1> result = zeros<T>(1);
    cuda::check_cublas_status(dot(handle, t1.size(), t1.raw_ptr(), 1, t2.raw_ptr(), 1, result.raw_ptr()), "Failed to perform dot product");
    return result;
};

/**
 * @brief Performs matrix multiplication between two 2D tensors.
 *
 * @tparam T: Data type of the tensors.
 * @param tensor1: The first tensor.
 * @param tensor2: The second tensor.
 * @param handle: The CuBLAS handle. The default is the global CuBLAS handle.
 * @return Tensor: The result tensor.
 */
template <typename T>
Tensor<T, 2> matmul(Tensor<T, 2>& tensor1, Tensor<T, 2>& tensor2, cublasHandle_t handle = cuda::cublas.get_handle()) {
    assert(tensor1.order() == vt::Order::C);
    assert(tensor2.order() == vt::Order::C);
    auto shape1 = tensor1.shape();
    auto shape2 = tensor2.shape();
    assert(shape1[1] == shape2[0]);

    auto t1 = ascontiguoustensor(tensor1);
    auto t2 = ascontiguoustensor(tensor2);
    auto alpha = T{1.0};
    auto beta = T{0.0};

    auto m = shape1[0];
    auto n = shape1[1];
    auto k = shape2[1];
    auto result = zeros<T>(m, k);

    auto gemm = cuda::CuBLASFunc<T>::gemm();
    cuda::check_cublas_status(gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, k, m, n, &alpha, t2.raw_ptr(), k, t1.raw_ptr(), n, &beta, result.raw_ptr(), k),
                              "Failed to perform matrix multiplication");

    return result;
}

/**
 * @brief Performs matrix multiplication between two N>2 tensors using batched matrix multiplication.
 * This is a specialized version of the matmul function for N>2 tensors.
 *
 * @tparam T: Data type of the tensors.
 * @param tensor1: The first tensor.
 * @param tensor2: The second tensor.
 * @param handle: The CuBLAS handle. The default is the global CuBLAS handle.
 * @return Tensor: The result tensor.
 */
template <typename T, size_t N>
Tensor<T, N> matmul(Tensor<T, N>& tensor1, Tensor<T, N>& tensor2, cublasHandle_t handle = cuda::cublas.get_handle()) {
    assert(tensor1.order() == vt::Order::C);
    assert(tensor2.order() == vt::Order::C);
    assert_at_least_3d_tensor<N>();
    auto shape1 = tensor1.shape();
    auto shape2 = tensor2.shape();
    assert(shape1[N - 1] == shape2[N - 2]);

    Shape<N> shape;
    int num_batches = 1;
    for (auto i = 0; i < N - 2; i++) {
        assert(shape1[i] == shape2[i]);
        num_batches *= shape1[i];
        shape[i] = shape1[i];
    }
    shape[N - 2] = shape1[N - 2];
    shape[N - 1] = shape2[N - 1];

    auto t1 = ascontiguoustensor(tensor1);
    auto t2 = ascontiguoustensor(tensor2);

    auto alpha = T{1.0};
    auto beta = T{0.0};
    auto m = shape1[N - 2];
    auto n = shape1[N - 1];
    auto k = shape2[N - 1];
    auto result = zeros<T>(shape);

    auto gemm = cuda::CuBLASFunc<T>::gemm_batched();
    cuda::check_cublas_status(gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, k, m, n, &alpha, t2.raw_ptr(), k, shape2[N - 1] * shape2[N - 2], t1.raw_ptr(), n,
                                   shape1[N - 1] * shape1[N - 2], &beta, result.raw_ptr(), k, shape[N - 1] * shape[N - 2], num_batches),
                              "Failed to perform batch matrix multiplication");

    return result;
}

}  // namespace vt
