#pragma once

#include <lib/core/tensor.hpp>

namespace vt {

/**
 * @brief Print the tensor. It is called iteratively to print the tensor for each row.
 *
 * @param tensor: The tensor object.
 * @param dim: The dimension of the tensor.
 * @param indices: The indices of the tensor.
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 */
template <typename T, size_t N>
void print_tensor(const Tensor<T, N>& tensor, size_t dim, Shape<N> indices) {
    auto start = tensor.start();
    auto shape = tensor.shape();
    auto strides = tensor.strides();
    auto data = tensor.data();
    if (dim == N - 1) {
        std::cout << "[";
        for (size_t i = 0; i < shape[dim]; ++i) {
            indices[dim] = i;
            size_t index = start;
            for (size_t j = 0; j < N; ++j) {
                index += indices[j] * strides[j];
            }
            std::cout << (*data)[index] << " ";
        }
        std::cout << "]" << std::endl;
    } else {
        for (size_t i = 0; i < shape[dim]; ++i) {
            indices[dim] = i;
            print_tensor(tensor, dim + 1, indices);
        }
    }
}

/**
 * @brief Print the tensor.
 *
 * @param tensor: The tensor object.
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 */
template <typename T, size_t N>
void print(const Tensor<T, N>& tensor) {
    std::cout << "Tensor = ";
    Shape<N> indices{};
    print_tensor(tensor, 0, indices);
}

}  // namespace vt
