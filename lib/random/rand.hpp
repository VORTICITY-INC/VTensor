#pragma once

#include <lib/core/tensor.hpp>
#include <lib/random/curand.hpp>

namespace vt {

namespace random {

/**
 * @brief Generate a tensor of random numbers from a uniform distribution.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param shape: Shape of the tensor.
 * @param gen: The CuRand generator. The default is the global CuRand generator.
 * @return Tensor<T, N>: The tensor of random numbers.
 */
template <typename T = float, size_t N>
Tensor<T, N> rand(Shape<N> shape, curandGenerator_t gen = cuda::curand.get_handle()) {
    auto tensor = Tensor<T, N>(shape);
    curandStatus_t status;
    if constexpr (std::is_same<T, float>::value) {
        status = curandGenerateUniform(gen, tensor.raw_ptr(), tensor.size());
    } else if constexpr (std::is_same<T, double>::value) {
        status = curandGenerateUniformDouble(gen, tensor.raw_ptr(), tensor.size());
    }
    cuda::check_curand_status(status, "Failed to generate normal random numbers");
    return tensor;
}

/**
 * @brief Generate a 1D tensor of random numbers from a uniform distribution.
 *
 * @tparam T: Data type of the tensor.
 * @param m: Size of the tensor.
 * @param gen: The CuRand generator. The default is the global CuRand generator.
 * @return Tensor<T, 1>: The tensor of random numbers.
 */
template <typename T = float>
Tensor<T, 1> rand(size_t m, curandGenerator_t gen = cuda::curand.get_handle()) {
    return rand<T, 1>({m}, gen);
}

/**
 * @brief Generate a 2D tensor of random numbers from a uniform distribution.
 *
 * @tparam T: Data type of the tensor.
 * @param m: Number of rows of the tensor.
 * @param n: Number of columns of the tensor.
 * @param gen: The CuRand generator. The default is the global CuRand generator.
 * @return Tensor<T, 2>: The tensor of random numbers.
 */
template <typename T = float>
Tensor<T, 2> rand(size_t m, size_t n, curandGenerator_t gen = cuda::curand.get_handle()) {
    return rand<T, 2>({m, n}, gen);
}

}  // namespace random

}  // namespace vt
