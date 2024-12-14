#pragma once

#include <cuda_runtime.h>

#include "lib/core/tensor.hpp"
#include "xtensor/xarray.hpp"

namespace vt {

/**
 * @brief Copy a tensor from device to host array.
 *
 * @param tensor: The tensor object.
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @return xt::xarray<T>: Host array.
 */
template <typename T, size_t N>
xt::xarray<T> asxarray(const Tensor<T, N>& tensor) {
    auto s = tensor.shape();
    std::vector<size_t> shape(s.begin(), s.end());
    xt::xarray<T> arr(shape);
    if constexpr (std::is_same_v<T, bool>) {
        thrust::copy(tensor.begin(), tensor.end(), arr.begin());
    } else {
        if (tensor.contiguous()) {
            auto s = tensor.size() * sizeof(T);
            cudaMemcpy(arr.data(), tensor.raw_ptr(), s, cudaMemcpyDeviceToHost);
        } else {
            thrust::copy(tensor.begin(), tensor.end(), arr.begin());
        }
    }
    return arr;
}

}  // namespace vt
