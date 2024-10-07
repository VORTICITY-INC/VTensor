#pragma once

#include <vector>

#include "cnpy.h"
#include "lib/core/tensor.hpp"
#include "lib/memory/astensor.hpp"
#include "lib/memory/asvector.hpp"

namespace vt {

/**
 * @brief Save tensor to a file.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param filename: Name of the file.
 * @param tensor: Tensor to be saved.
 */
template <typename T, size_t N>
void save(const std::string& filename, const Tensor<T, N>& tensor) {
    auto vec = asvector(tensor);
    auto _shape = tensor.shape();
    std::vector<size_t> shape(_shape.begin(), _shape.end());
    cnpy::npy_save(filename, &vec[0], shape, "w");
}

/**
 * @brief Load tensor from a file.
 *
 * @tparam T: Data type of the tensor.
 * @param filename: Name of the file.
 * @return Tensor<T, 1>: The tensor loaded from the file.
 */
template <typename T>
Tensor<T, 1> load(const std::string& filename) {
    auto arr = cnpy::npy_load(filename);
    return astensor(arr.data<T>(), arr.num_vals);
}

}  // namespace vt
