
#pragma once

namespace vt {

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

}  // namespace vt
