#pragma once

#include "lib/core/tensor.hpp"
#include "lib/generator/zeros.hpp"
#include "lib/linalg/cublas.hpp"
#include "lib/memory/ascontiguoustensor.hpp"

namespace vt {

/**
 * @brief Convert a 2D tensor to a Fortran ordered tensor.
 * We use the CuBLAS geam function to transpose the 2D tensor.
 * [TODO] For higher dimensions, we will copy the tensor elementwise.
 *
 * @tparam T: Data type of the tensor.
 * @param tensor: The tensor object.
 * @return Tensor: The Fortran tensor object.
 */
template <typename T>
Tensor<T, 2> asfortrantensor(Tensor<T, 2>& tensor) {
    auto ctensor = ascontiguoustensor(tensor);
    auto shape = ctensor.shape();
    int m = shape[0];
    int n = shape[1];
    auto result = zeros<T>(shape, Order::F);
    auto handle = cuda::CuBLAS::get_instance().get_handle();
    auto alpha = T{1.0};
    auto beta = T{0.0};
    auto geam = cuda::CuBLASFunc<T>::geam();
    auto strides = Shape<2>{1, shape[0]};
    geam(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, &alpha, ctensor.raw_ptr(), n, &beta, ctensor.raw_ptr(), n, result.raw_ptr(), m);
    result.set_strides(strides);  // Need to update the strides of the tensor.
    return result;
}

}  // namespace vt