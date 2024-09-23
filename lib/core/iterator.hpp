#pragma once

#include <thrust/device_vector.h>

namespace vt {

/**
 * @brief Get the iterator index.
 *
 * @tparam N: Number of dimensions.
 * @param n: The index.
 * @param shape: The shape of the tensor.
 * @param strides: The strides of the tensor.
 * @param start: The start index of the tensor.
 * @return size_t: The index of the iterator.
 */
template <size_t N>
__host__ __device__ size_t get_iterator_index(size_t n, const size_t* shape, const size_t* strides, size_t start) {
    size_t index = start;
    for (size_t i = 0; i < N; ++i) {
        size_t shape_product = 1;
        for (size_t j = i + 1; j < N; ++j) shape_product *= shape[j];
        index += (n / shape_product) * strides[i];
        n %= shape_product;
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
     * @param contiguous: The contiguous flag of the tensor.
     */
    __host__ __device__ TensorIterator(const Iterator& iter, const size_t* shape, const size_t* strides, const size_t start, const bool contiguous)
        : thrust::iterator_adaptor<TensorIterator<Iterator, N>, Iterator>(iter), start(start), contiguous(contiguous) {
        for (size_t i = 0; i < N; ++i) {
            this->shape[i] = shape[i];
            this->strides[i] = strides[i];
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
            this->base_reference() += get_iterator_index<N>(n, shape, strides, start);
        }
    }

   private:
    size_t shape[N];
    size_t strides[N];
    size_t start;
    bool contiguous;
};

}  // namespace vt
