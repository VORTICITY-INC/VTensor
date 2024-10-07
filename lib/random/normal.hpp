#pragma once

#include "lib/core/tensor.hpp"
#include "lib/random/curand.hpp"

namespace vt {

namespace random {

/**
 * @brief Generate a tensor of random numbers from a normal distribution.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param shape: Shape of the tensor.
 * @param mean: Mean of the normal distribution.
 * @param stddev: Standard deviation of the normal distribution.
 * @param gen: The CuRand generator. The default is the global CuRand generator.
 * @return Tensor<T, N>: The tensor of random numbers.
 */
template <typename T = float, size_t N>
Tensor<T, N> normal(Shape<N> shape, T mean = T{0.0}, T stddev = T{1.0}, curandGenerator_t gen = cuda::curand.get_handle()) {
    auto tensor = Tensor<T, N>(shape);
    curandStatus_t status;
    if constexpr (std::is_same<T, float>::value) {
        status = curandGenerateNormal(gen, tensor.raw_ptr(), tensor.size(), mean, stddev);
    } else if constexpr (std::is_same<T, double>::value) {
        status = curandGenerateNormalDouble(gen, tensor.raw_ptr(), tensor.size(), mean, stddev);
    }
    cuda::check_curand_status(status, "Failed to generate normal random numbers");
    return tensor;
}

/**
 * @brief Generate a 1D tensor of random numbers from a normal distribution.
 *
 * @tparam T: Data type of the tensor.
 * @param m: Size of the tensor.
 * @param mean: Mean of the normal distribution.
 * @param stddev: Standard deviation of the normal distribution.
 * @param gen: The CuRand generator. The default is the global CuRand generator.
 * @return Tensor<T, 1>: The tensor of random numbers.
 */
template <typename T = float>
Tensor<T, 1> normal(size_t m, T mean = T{0.0}, T stddev = T{1.0}, curandGenerator_t gen = cuda::curand.get_handle()) {
    auto shape = Shape<1>{m};
    return normal<T, 1>(shape, mean, stddev, gen);
}

/**
 * @brief Generate a 2D tensor of random numbers from a normal distribution.
 *
 * @tparam T: Data type of the tensor.
 * @param m: Number of rows of the tensor.
 * @param n: Number of columns of the tensor.
 * @param mean: Mean of the normal distribution.
 * @param stddev: Standard deviation of the normal distribution.
 * @param gen: The CuRand generator. The default is the global CuRand generator.
 * @return Tensor<T, 2>: The tensor of random numbers.
 */
template <typename T = float>
Tensor<T, 2> normal(size_t m, size_t n, T mean = T{0.0}, T stddev = T{1.0}, curandGenerator_t gen = cuda::curand.get_handle()) {
    auto shape = Shape<2>{m, n};
    return normal<T, 2>(shape, mean, stddev, gen);
}

}  // namespace random

}  // namespace vt
