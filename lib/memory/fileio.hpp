#pragma once

#include <vector>

#include "lib/core/tensor.hpp"
#include "lib/memory/astensor.hpp"
#include "lib/memory/asxarray.hpp"
#include "xtensor/xnpy.hpp"

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
    auto arr = asxarray(tensor);
    xt::dump_npy(filename, arr);
}

/**
 * @brief Load tensor from a file.
 *
 * @tparam T: Data type of the tensor.
 * @param filename: Name of the file.
 * @return Tensor<T, 1>: The tensor loaded from the file.
 */
template <typename T, size_t N>
Tensor<T, N> load(const std::string& filename) {
    auto arr = xt::load_npy<T>(filename);
    return astensor<T, N>(arr);
}

}  // namespace vt
