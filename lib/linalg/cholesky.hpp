#pragma once

#include <lib/core/assertions.hpp>
#include <lib/core/tensor.hpp>
#include <lib/generator/zeros.hpp>
#include <lib/linalg/cusolver.hpp>
#include <lib/memory/copy.hpp>

namespace vt {

namespace linalg {

/**
 * @brief Perform Cholesky decomposition on a 2D tensor.
 *
 * @tparam T: Data type of the tensor.
 * @param tensor: The tensor object to be decomposed.
 * @param handle: The CuSolver handle. The default is the global CuSolver handle.
 * @return Tensor: The lower triangle of the Cholesky decomposition.
 */
template <typename T>
Tensor<T, 2> cholesky(Tensor<T, 2>& tensor, cusolverDnHandle_t handle = cuda::cusolver.get_handle()) {
    auto [n, m] = tensor.shape();
    assert(n == m);
    auto x = copy(tensor);

    // Get buffer size
    int lwork;
    auto helper = cuda::CuSolverFunc<T>::potrf_buffer_size();
    cuda::check_cusolver_status(helper(handle, CUBLAS_FILL_MODE_UPPER, n, x.raw_ptr(), n, &lwork), "Failed to get buffer size");

    // Perform Cholesky decomposition
    auto dev_info = zeros<int>(1);
    auto workspace = zeros<T>(lwork);
    auto potrf = cuda::CuSolverFunc<T>::potrf();
    cuda::check_cusolver_status(potrf(handle, CUBLAS_FILL_MODE_UPPER, n, x.raw_ptr(), n, workspace.raw_ptr(), lwork, dev_info.raw_ptr()),
                                "Failed to perform Cholesky decomposition");

    // Lower triangle
    tril(x, 0, false);
    return x;
}

/**
 * @brief Perform batched Cholesky decomposition on a N-D tensor.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param tensor: The tensor object to be decomposed.
 * @param handle: The CuSolver handle. The default is the global CuSolver handle.
 * @return Tensor: The lower triangle of the Cholesky decomposition.
 */
template <typename T, size_t N>
Tensor<T, N> cholesky(Tensor<T, N>& tensor, cusolverDnHandle_t handle = cuda::cusolver.get_handle()) {
    assert_at_least_3d_tensor<N>();
    auto shape = tensor.shape();
    assert(shape[N - 1] == shape[N - 2]);
    Shape<N - 2> batch_shape;
    std::copy(shape.begin(), shape.begin() + (N - 2), batch_shape.begin());
    size_t batch_size = get_size(batch_shape);
    auto x = copy(tensor);
    auto n = shape[N - 1];
    auto ldx = tensor.strides()[N - 2];
    auto dev_info = zeros<int>(batch_size);
    auto ptrs = arange<T>(x.raw_ptr(), x.raw_ptr() + n * n * batch_size, n * n);
    auto potrf_batched = cuda::CuSolverFunc<T>::potrf_batched();
    cuda::check_cusolver_status(potrf_batched(handle, CUBLAS_FILL_MODE_UPPER, n, ptrs.raw_ptr(), ldx, dev_info.raw_ptr(), batch_size),
                                "Failed to perform Cholesky decomposition");
    return tril(x);
}

}  // namespace linalg

}  // namespace vt
