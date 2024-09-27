#pragma once

#include <lib/core/tensor.hpp>
#include <lib/generator/zeros.hpp>
#include <lib/linalg/cusolver.hpp>
#include <lib/memory/copy.hpp>


namespace vt {

namespace linalg {

template <typename T>
Tensor<T, 2> cholesky(Tensor<T, 2>& tensor, cusolverDnHandle_t handle = cuda::cusolver.get_handle()) {
    auto [n, m] = tensor.shape();
    assert(n == m);
    auto x = copy(tensor);
    auto dev_info = zeros<int>(1);

    int lwork;
    auto helper = cuda::CuSolverFunc<T>::potrf_buffer_size();
    helper(handle, CUBLAS_FILL_MODE_UPPER, n, x.raw_ptr(), n, &lwork);

    auto workspace = zeros<T>(lwork);
    auto potrf = cuda::CuSolverFunc<T>::potrf();

    potrf(handle, CUBLAS_FILL_MODE_UPPER, n, x.raw_ptr(), n, workspace.raw_ptr(), lwork, dev_info.raw_ptr());
    tril(x, 0, false);
    return x;
}

template <typename T, size_t N>
Tensor<T, N> cholesky(Tensor<T, N>& tensor, cusolverDnHandle_t handle = cuda::cusolver.get_handle()) {
    auto shape = tensor.shape();
    assert(shape[N-1] == shape[N-2]);
    Shape<N - 2> batch_shape;
    std::copy(shape.begin(), shape.begin() + (N - 2), batch_shape.begin());
    size_t batch_size = get_size(batch_shape);
    auto x = copy(tensor);
    auto n = shape[N-1];
    auto ldx = tensor.strides()[N-2];
    auto dev_info = zeros<int>(batch_size);
    auto ptrs = arange<T>(x.raw_ptr(), x.raw_ptr() + n * n * batch_size, n * n);
    auto potrf_batched = cuda::CuSolverFunc<T>::potrf_batched();
    potrf_batched(handle, CUBLAS_FILL_MODE_UPPER, n, ptrs.raw_ptr(), ldx, dev_info.raw_ptr(), batch_size);
    return tril(x);
}


}  // namespace linalg

}  // namespace vt
