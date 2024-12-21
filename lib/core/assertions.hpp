
#pragma once

#include <cassert>

namespace vt {

enum class Order : char;

/**
 * @brief Assert that the tensor is at least 1D.
 *
 * @tparam N: Number of dimensions.
 */
template <size_t N>
constexpr void assert_at_least_1d_tensor() {
    static_assert(N > 0, "The tensor must be at least 1D.");
}

/**
 * @brief Assert that the tensor is at least 2D.
 *
 * @tparam N: Number of dimensions.
 */
template <size_t N>
constexpr void assert_at_least_2d_tensor() {
    static_assert(N > 1, "The tensor must be at least 2D.");
}

/**
 * @brief Assert that the tensor is at least 3D.
 *
 * @tparam N: Number of dimensions.
 */
template <size_t N>
constexpr void assert_at_least_3d_tensor() {
    static_assert(N > 2, "The tensor must be at least 3D.");
}

/**
 * @brief Assert same memory layout order for two tensors.
 *
 * @param order1: The order of the first tensor.
 * @param order2: The order of the second tensor.
 */
inline void assert_same_order_between_two_tensors(const Order order1, const Order order2) { assert(order1 == order2); }

}  // namespace vt
