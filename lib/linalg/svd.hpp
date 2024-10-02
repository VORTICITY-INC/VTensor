#pragma once

#include <lib/core/assertions.hpp>
#include <lib/core/tensor.hpp>
#include <lib/generator/eye.hpp>
#include <lib/generator/zeros.hpp>
#include <lib/linalg/cublas.hpp>
#include <lib/linalg/cusolver.hpp>
#include <lib/memory/copy.hpp>
#include <tuple>

namespace vt {

namespace linalg {

/**
 * @brief Singular Value Decomposition.
 *
 * @tparam T: Data type of the tensor.
 * @param tensor: The tensor object to be decomposed.
 * @param full_matrices: If True, it returns u and v with full dimensions.
 * @param compute_uv: If False, it only returns singular values.
 * @param handle: The CuSolver handle. The default is the global CuSolver handle.
 * @return A tuple of U, S, V, where tensor = U * diag(S) * V. (diag neeeded to be implemented)
 */
template <typename T>
std::tuple<Tensor<T, 2>, Tensor<T, 1>, Tensor<T, 2>> svd(Tensor<T, 2>& tensor, bool full_matrices = true, bool compute_uv = true,
                                                         cusolverDnHandle_t handle = cuda::cusolver.get_handle()) {
    auto [n, m] = tensor.shape();
    Tensor<T, 2> x;
    bool trans_flag;

    // CuSolver only supports m >= n for SVD. If m < n, we need to transpose the tensor.
    if (m >= n) {
        x = copy(tensor);
        trans_flag = false;
    } else {
        x = copy(transpose(tensor));
        std::swap(n, m);
        trans_flag = true;
    }

    // Caculate the buffer size
    int buffer_size;
    auto helper = cuda::CuSolverFunc<T>::gesvd_buffer_size();
    cuda::check_cusolver_status(helper(handle, m, n, &buffer_size), "Failed to get buffer size");
    auto buffer = zeros<T>(buffer_size);

    // Initialize U and VT
    Tensor<T, 2> u;
    Tensor<T, 2> vt;

    signed char job_u;
    signed char job_vt;

    T* u_ptr;
    T* vt_ptr;

    if (compute_uv) {
        if (full_matrices) {
            u = zeros<T>(m, m);
            vt = x({0, n}, {0, n});
            job_u = 'A';
            job_vt = 'O';
        } else {
            u = x;
            vt = zeros<T>(n, n);
            job_u = 'O';
            job_vt = 'S';
        }
        u_ptr = u.raw_ptr();
        vt_ptr = vt.raw_ptr();

    } else {
        u_ptr = nullptr;
        vt_ptr = nullptr;
        job_u = 'N';
        job_vt = 'N';
    }

    // Perform SVD
    auto dev_info = zeros<int>(1);
    auto s = zeros<T>(n);
    auto rwork = zeros<T>(n - 1);
    auto gesvd = cuda::CuSolverFunc<T>::gesvd();
    cuda::check_cusolver_status(gesvd(handle, job_u, job_vt, m, n, x.raw_ptr(), m, s.raw_ptr(), u_ptr, m, vt_ptr, n, buffer.raw_ptr(), buffer_size,
                                      rwork.raw_ptr(), dev_info.raw_ptr()),
                                "Failed to perform SVD");

    if (trans_flag) {
        return std::make_tuple(transpose(u), s, transpose(vt));
    } else {
        return std::make_tuple(vt, s, u);
    }
}

/**
 * @brief Batched Singular Value Decomposition.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param tensor: The tensor object to be decomposed.
 * @param full_matrices: If True, it returns u and v with full dimensions.
 * @param compute_uv: If False, it only returns singular values.
 * @param handle: The CuSolver handle. The default is the global CuSolver handle.
 * @return A tuple of U, S, V, where tensor = U * diag(S) * V. (diag neeeded to be implemented)
 */
template <typename T, size_t N>
std::tuple<Tensor<T, N>, Tensor<T, N - 1>, Tensor<T, N>> svd(Tensor<T, N>& tensor, bool full_matrices = true, bool compute_uv = true,
                                                             cusolverDnHandle_t handle = cuda::cusolver.get_handle()) {
    assert_at_least_3d_tensor<N>();
    auto shape = tensor.shape();
    Shape<N - 2> batch_shape;
    std::copy(shape.begin(), shape.begin() + (N - 2), batch_shape.begin());
    size_t batch_size = get_size(batch_shape);

    auto m = shape[N - 2];
    auto n = shape[N - 1];
    auto a = copy(moveaxis(tensor, N - 2, N - 1));

    auto lda = m;
    auto mn = std::min(m, n);
    auto s = zeros<T>(expand_shape(batch_shape, mn));
    auto ldu = m;
    auto ldv = n;

    // Get buffer size and info.
    cusolverEigMode_t jobz = compute_uv ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;
    auto u = moveaxis(zeros<T>(expand_shape(batch_shape, m, ldu)), N - 2, N - 1);
    auto v = moveaxis(zeros<T>(expand_shape(batch_shape, n, ldv)), N - 2, N - 1);
    auto helper = cuda::CuSolverFunc<T>::gesvdj_batched_buffer_size();
    gesvdjInfo_t params;
    int lwork;
    cuda::check_cusolver_status(cusolverDnCreateGesvdjInfo(&params), "Failed to create gesvdj info");
    cuda::check_cusolver_status(helper(handle, jobz, m, n, a.raw_ptr(), lda, s.raw_ptr(), u.raw_ptr(), ldu, v.raw_ptr(), ldv, &lwork, params, batch_size),
                                "Failed to get buffer size");

    // Perform SVD
    auto work = zeros<T>(lwork);
    auto info = zeros<int>(batch_size);

    auto gesvdj = cuda::CuSolverFunc<T>::gesvdj_batched();
    cuda::check_cusolver_status(gesvdj(handle, jobz, m, n, a.raw_ptr(), lda, s.raw_ptr(), u.raw_ptr(), ldu, v.raw_ptr(), ldv, work.raw_ptr(), lwork,
                                       info.raw_ptr(), params, batch_size),
                                "Failed to perform gesvdj");

    cuda::check_cusolver_status(cusolverDnDestroyGesvdjInfo(params), "Failed to destroy gesvdj info");

    if (!full_matrices) {
        u = u(ellipsis, {0, mn});
        v = v(ellipsis, {0, mn});
    }
    v = moveaxis(v, N - 2, N - 1);
    return std::make_tuple(u, s, v);
}

}  // namespace linalg

}  // namespace vt