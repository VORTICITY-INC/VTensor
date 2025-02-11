#pragma once

#include <vector>

#include "lib/core/tensor.hpp"
#include "lib/generator/zeros.hpp"
#include "xtensor/xarray.hpp"

namespace vt {

/**
 * @brief Copy std::vector to the tensor.
 *
 * @tparam T: Data type of the tensor.
 * @param vector: std::vector object.
 * @param order: Order of the tensor.
 * @return Tensor: The tensor from std::vector.
 */
template <typename T>
Tensor<T, 1> astensor(const std::vector<T>& vector, const Order order = Order::C) {
    auto tensor = zeros<T>(vector.size(), order);
    if constexpr (std::is_same_v<T, bool>) {
        thrust::copy(vector.begin(), vector.end(), tensor.begin());
    } else {
        auto size = tensor.size() * sizeof(T);
        cudaMemcpy(tensor.raw_ptr(), vector.data(), size, cudaMemcpyHostToDevice);
    }
    return tensor;
}

/**
 * @brief Copy xarray to the tensor.
 *
 * @tparam T: Data type of the tensor.
 * @tparam L: Layout type of the xarray.
 * @tparam N: Number of dimensions of the tensor.
 * @param arr: The xarray object.
 * @return Tensor: The tensor from the xarray.
 */
template <typename T, size_t N, typename C>
Tensor<T, N> astensor(const C& arr) {
    Order order;
    const auto L = arr.layout();
    if (L == xt::layout_type::row_major) {
        order = Order::C;
    } else if (L == xt::layout_type::column_major) {
        order = Order::F;
    } 

    auto sh = arr.shape();
    vt::Shape<N> shape;
    assert(sh.size() == N);
    std::copy(sh.begin(), sh.end(), shape.begin());
    auto tensor = zeros<T>(shape, order);
    if constexpr (std::is_same_v<T, bool>) {
        thrust::copy(arr.begin(), arr.end(), tensor.begin());
    } else {
        auto size = tensor.size() * sizeof(T);
        cudaMemcpy(tensor.raw_ptr(), arr.data(), size, cudaMemcpyHostToDevice);
    }
    return tensor;
}

/**
 * @brief Copy the raw pointer to the tensor.
 *
 * @tparam T: Data type of the tensor.
 * @param ptr: The raw pointer.
 * @param size: The size of the tensor.
 * @param order: Order of the tensor.
 * @return Tensor: The tensor from the raw pointer.
 */
template <typename T>
Tensor<T, 1> astensor(const T* ptr, const size_t size, const Order order = Order::C) {
    auto tensor = zeros<T>(size, order);
    if constexpr (std::is_same_v<T, bool>) {
        thrust::copy(ptr, ptr + size, tensor.begin());
    } else {
        auto s = tensor.size() * sizeof(T);
        cudaMemcpy(tensor.raw_ptr(), ptr, s, cudaMemcpyHostToDevice);
    }
    return tensor;
}

/**
 * @brief Copy the raw pointer to the tensor.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @param ptr: The raw pointer.
 * @param shape: The shape of the tensor.
 * @param order: Order of the tensor.
 * @return Tensor: The tensor from the raw pointer.
 */
template <typename T, size_t N>
Tensor<T, N> astensor(const T* ptr, const Shape<N> shape, const Order order = Order::C) {
    auto tensor = zeros<T, N>(shape, order);
    if constexpr (std::is_same_v<T, bool>) {
        thrust::copy(ptr, ptr + tensor.size(), tensor.begin());
    } else {
        auto s = tensor.size() * sizeof(T);
        cudaMemcpy(tensor.raw_ptr(), ptr, s, cudaMemcpyHostToDevice);
    }
    return tensor;
}

}  // namespace vt