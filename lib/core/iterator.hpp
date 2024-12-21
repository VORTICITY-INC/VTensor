#pragma once

#include <thrust/device_vector.h>

namespace vt {

/**
 * @brief Order of the tensor.
 */
enum class Order : char { C, F };

/**
 * @brief Get the iterator index.
 *
 * @tparam N: Number of dimensions.
 * @param n: The index.
 * @param shape: The shape of the tensor.
 * @param strides: The strides of the tensor.
 * @param start: The start index of the tensor.
 * @param order: The order of the tensor.
 */
template <size_t N>
__host__ __device__ size_t get_iterator_index(size_t n, const size_t* shape, const size_t* strides, size_t start, Order order) {
    size_t index = start;
    if constexpr (N > 0) {
        if (order == Order::C) {
            for (size_t i = 0; i < N; ++i) {
                size_t shape_product = 1;
                for (size_t j = i + 1; j < N; ++j) shape_product *= shape[j];
                index += (n / shape_product) * strides[i];
                n %= shape_product;
            }
        } else {
            for (size_t i = N; i-- > 0;) {
                size_t shape_product = 1;
                for (size_t j = 0; j < i; ++j) shape_product *= shape[j];
                index += (n / shape_product) * strides[i];
                n %= shape_product;
            }
        }
    }
    return index;
}

/**
 * @brief TensorIterator class to represent an iterator of a tensor.
 *
 * @tparam Iterator: The raw iterator.
 * @tparam N: Number of dimensions of the tensor.
 */
template <typename Iterator, size_t N>
class TensorIterator : public thrust::iterator_adaptor<TensorIterator<Iterator, N>, Iterator> {
   public:
    /**
     * @brief Construct a new tensor Iterator object.
     *
     * @param iter: The raw iterator from thrust.
     * @param shape: The shape of the tensor.
     * @param strides: The strides of the tensor.
     * @param start: The start index of the tensor.
     * @param order: The order of the tensor.
     * @param contiguous: The contiguous flag of the tensor.
     */
    __host__ __device__ TensorIterator(const Iterator& iter, const size_t* shape, const size_t* strides, const size_t start, const Order order,
                                       const bool contiguous)
        : thrust::iterator_adaptor<TensorIterator<Iterator, N>, Iterator>(iter), start(start), order(order), contiguous(contiguous) {
        if constexpr (N > 0) {
            for (size_t i = 0; i < N; ++i) {
                this->shape[i] = shape[i];
                this->strides[i] = strides[i];
            }
        }
    }

    /**
     * @brief Advance the iterator by n steps.
     * If the iterator is contiguous, used the default advance method, otherwise calculate the new index based on the strides and shape.
     * @param n: Number of steps.
     */
    __host__ __device__ void advance(typename thrust::iterator_difference<Iterator>::type n) {
        if (contiguous) {
            this->base_reference() += n;
        } else {
            this->base_reference() += get_iterator_index<N>(n, shape, strides, start, order);
        }
    }

   private:
    size_t shape[N];
    size_t strides[N];
    size_t start;
    Order order;
    bool contiguous;
};

}  // namespace vt
