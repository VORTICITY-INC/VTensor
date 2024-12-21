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
    auto order = tensor.order();
    if (order == Order::C) {
        xt::dump_npy(filename, asxarray<T, N, xt::layout_type::row_major>(tensor));
    } else {
        xt::dump_npy(filename, asxarray<T, N, xt::layout_type::column_major>(tensor));
    }
}

/**
 * @brief Load tensor from a file.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @tparam L: Layout type of the tensor.
 * @param filename: Name of the file.
 * @return Tensor<T, N>: The tensor loaded from the file.
 */
template <typename T, size_t N, xt::layout_type L>
Tensor<T, N> load(const std::string& filename) {
    return astensor<T, N, L>(xt::load_npy<T>(filename));
}

}  // namespace vt
