#pragma once

#include <lib/core/tensor.hpp>

namespace vt {

/**
 * @brief Slice class to represent a slice of a tensor.
 * Usage:
 * Slice(0) -> Slice from 0 to 1 with step 1
 * Slice(0, 10) -> Slice from 0 to 10 with step 1
 * Slice(0, 10, 2) -> Slice from 0 to 10 with step 2
 */
class Slice {
   public:
    size_t start;
    size_t end;
    size_t step;

    // Default constructor
    Slice() = default;

    /**
     * @brief Construct a new Slice object
     *
     * @param start: The start index of the slice.
     */
    Slice(size_t start) : start(start), end(start + 1), step(1) {}

    /**
     * @brief Construct a new Slice object
     *
     * @param start: The start index of the slice.
     * @param end: The end index of the slice.
     */
    Slice(size_t start, size_t end) : start(start), end(end), step(1) {}

    /**
     * @brief Construct a new Slice object
     *
     * @param start: The start index of the slice.
     * @param end: The end index of the slice.
     * @param step: The step of the slice.
     */
    Slice(size_t start, size_t end, size_t step) : start(start), end(end), step(step) {}
};

/// Forward declaration of the Tensor class.
template <typename T, size_t N>
class Tensor;

/**
 * @brief TensorSliceProxy to perform a copy assignment operation for a sliced tensor.
 *
 * It helps to distingush between the Tensor copy assignment operator with TensorSliceProxy copy assignment operator.
 * tensor1 = tensor2; -> Tensor copy assignment operator, the content in tensor1 will be overwritten by tensor2.
 * tensor1(Slice(0, 2)) = tensor2; -> TensorSliceProxy copy assignment operator, the content in tensor1 would not be overwritten by tensor2.
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 */
template <typename T, size_t N>
class TensorSliceProxy : public Tensor<T, N> {
   public:
    /**
     * @brief Construct a TensorSliceProxy object
     *
     * @param tensor: The sliced tensor.
     */
    TensorSliceProxy(const Tensor<T, N>& tensor) : Tensor<T, N>(tensor) {}

    /**
     * @brief Copy assignment operator for TensorSliceProxy from other TensorSliceProxy.
     *
     * @param other: The TensorSliceProxy to be copied.
     * @return TensorSliceProxy: The tensor slice proxy object.
     */
    TensorSliceProxy& operator=(const Tensor<T, N>& other) {
        assert(this->shape() == other.shape());
        thrust::copy(other.begin(), other.end(), this->begin());
        return *this;
    }

    /**
     * @brief Copy assignment operator for TensorSliceProxy from other TensorSliceProxy.
     *
     * @param other: The TensorSliceProxy to be copied.
     * @return TensorSliceProxy: The tensor slice proxy object.
     */
    TensorSliceProxy& operator=(const TensorSliceProxy<T, N>& other) { return operator=(Tensor<T, N>(other)); }

    /**
     * @brief Copy assignment operator for TensorSliceProxy from a constant.
     *
     * @param other: The value to be copied.
     * @return TensorSliceProxy: The tensor slice proxy object.
     */
    TensorSliceProxy& operator=(const T value) {
        thrust::fill(this->begin(), this->end(), value);
        return *this;
    }
};

}  // namespace vt
