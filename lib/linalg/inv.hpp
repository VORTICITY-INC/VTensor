#pragma once

#include <lib/core/tensor.hpp>
#include <lib/generator/eye.hpp>
#include <lib/generator/zeros.hpp>
#include <lib/linalg/cublas.hpp>
#include <lib/linalg/cusolver.hpp>
#include <lib/memory/copy.hpp>

namespace vt {

namespace linalg {

/**
 * @brief Compute the inverse of a 2D tensor.
 * This method is based on CuSolver LU factorization and solve. Notice that the input tensor is copied to avoid overwriting.
 *
 * @tparam T: Data type of the tensor.
 * @param tensor: The tensor object to be inversed.
 * @param handle: The CuSolver handle. The default is the global CuSolver handle.
 * @return Tensor: The inverse tensor object.
 */
template <typename T>
Tensor<T, 2> inv(Tensor<T, 2>& tensor, cusolverDnHandle_t handle = cuda::cusolver.get_handle()) {
    auto shape = tensor.shape();
    assert(shape[0] == shape[1]);

    // Copy the tensor, so it is not overwritten
    auto ctensor = copy(tensor);

    // Get the matrix dimension
    int n = shape[0];

    // Caculate the buffer size
    int buffer_size;
    auto helper = cuda::CuSolverFunc<T>::getrf_buffer_size();
    cuda::check_cusolver_status(helper(handle, n, n, ctensor.raw_ptr(), n, &buffer_size), "Failed to get buffer size");
    auto buffer = zeros<T>(buffer_size);

    // Perform LU factorization
    auto rf = cuda::CuSolverFunc<T>::getrf();
    auto pivot = zeros<int>(n);
    auto info = zeros<int>(1);
    cuda::check_cusolver_status(rf(handle, n, n, ctensor.raw_ptr(), n, buffer.raw_ptr(), pivot.raw_ptr(), info.raw_ptr()),
                                "Failed to perform LU factorization");

    // Solve Ax = I
    auto rs = cuda::CuSolverFunc<T>::getrs();
    auto result = eye<T>(n);
    cuda::check_cusolver_status(rs(handle, CUBLAS_OP_N, n, n, ctensor.raw_ptr(), n, pivot.raw_ptr(), result.raw_ptr(), n, info.raw_ptr()), "Failed to solve");

    return result;
}

/**
 * @brief Compute the inverse of a N-D tensor.
 * This method is based on CuBLAS LU factorization and solve. Notice that the input tensor is copied to avoid overwriting.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param tensor: The tensor object to be inversed.
 * @param handle: The CuBLAS handle. The default is the global CuBLAS handle.
 * @return Tensor<T, N>: The inverse tensor object.
 */
template <typename T, size_t N>
Tensor<T, N> inv(Tensor<T, N>& tensor, cublasHandle_t handle = cuda::cublas.get_handle()) {
    static_assert(N > 1, "The tensor must be at least 2D.");
    auto shape = tensor.shape();
    assert(shape[N - 1] == shape[N - 2]);

    // Copy the tensor, so it is not overwritten
    auto ctensor = copy(tensor);

    // Calculate the batch size
    int batch_size = 1;
    for (auto i = 0; i < N - 2; i++) {
        batch_size *= shape[i];
    }

    // Get the matrix dimension
    int n = shape[N - 1];

    // Perform LU factorization
    auto rf = cuda::CuBLASFunc<T>::getrf_batched();
    auto ctensor_arr = arange<T>(ctensor.raw_ptr(), ctensor.raw_ptr() + n * n * batch_size, n * n);
    auto pivot = zeros<int>(batch_size, n);
    auto info = zeros<int>(batch_size);
    cuda::check_cublas_status(rf(handle, n, ctensor_arr.raw_ptr(), n, pivot.raw_ptr(), info.raw_ptr(), batch_size), "Failed to perform LU factorization");

    // Solve and save the result in the same tensor
    auto ri = cuda::CuBLASFunc<T>::getri_batched();
    cuda::check_cublas_status(ri(handle, n, (const T**)ctensor_arr.raw_ptr(), n, pivot.raw_ptr(), ctensor_arr.raw_ptr(), n, info.raw_ptr(), batch_size),
                              "Failed to solve");

    return ctensor;
}

}  // namespace linalg

}  // namespace vt
