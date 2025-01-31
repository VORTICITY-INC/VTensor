#pragma once

#include "lib/core/assertions.hpp"
#include "lib/core/broadcast.hpp"
#include "lib/core/tensor.hpp"

namespace vt {

/**
 * @brief Slice type.
 */
enum class SliceType { Normal, All };

/**
 * @brief Slice class to represent a slice of a tensor.
 * Usage:
 * Slice(0) -> Slice from 0 to 1 with step 1
 * Slice(0, 10) -> Slice from 0 to 10 with step 1
 * Slice(0, 10, 2) -> Slice from 0 to 10 with step 2
 * Slice::all() -> Slice to represent the full range
 */
class Slice {
   public:
    size_t start;
    size_t end;
    size_t step;
    SliceType type = SliceType::Normal;

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

    /**
     * @brief Construct a new Slice object
     *
     * @param type: The type of the slice.
     */
    Slice(SliceType type) : type(type) {}

    /**
     * @brief Return a slice object to represent the full range.
     *
     * @return Slice: The slice object.
     */
    static Slice all() { return Slice(SliceType::All); }
};

/**
 * @brief Represents an ellipsis in tensor operations.
 *
 * The EllipsisT struct is used to signify an ellipsis in tensor slicing operations.
 * A global static object of this type, named `ellipsis`, is defined for convenience.
 */
struct EllipsisT {};

/// Ellipsis object
static constexpr EllipsisT ellipsis = {};

/**
 * @brief Represents an newaxis in tensor operations.
 *
 * The NewAxisT struct is used to signify a new axis in tensor operations.
 * A global static object of this type, named `newaxis`, is defined for convenience.
 */
struct NewAxisT {};

/// Newaxis object
static constexpr NewAxisT newaxis = {};

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
        assert_same_order_between_two_tensors(this->order(), other.order());
        auto [lhs, rhs] = broadcast(*this, other);
        thrust::copy(rhs.begin(), rhs.end(), lhs.begin());
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

/**
 * @brief AssignValue struct to assign a value to the tensor based on the condition tensor.
 *
 * @tparam T: Data type of the tensor.
 */
template <typename T>
struct AssignValue {
    /**
     * @brief Construct a new AssignValue object
     *
     * @param value: The value to be assigned.
     */
    AssignValue(T value) : value(value) {}

    /**
     * @brief Operator to assign a value to the tensor based on the condition.
     *
     * @param x: The value.
     * @return T: The value to be assigned.
     */
    __device__ T operator()(const T& x) const { return value; }

    T value;
};

/**
 * @brief TensorCondProxy to perform a copy assignment operator based on the condition.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 */
template <typename T, size_t N>
class TensorCondProxy : public Tensor<T, N> {
   public:
    /**
     * @brief Construct a TensorCondProxy object
     *
     * @param tensor: The tensor.
     * @param cond: The condition tensor.
     */
    TensorCondProxy(const Tensor<T, N>& tensor, const Tensor<bool, N>& cond) : Tensor<T, N>(tensor), cond(cond) {
        assert_same_order_between_two_tensors(this->order(), cond.order());
    }

    /**
     * @brief Copy assignment operator for the TensorCondProxy from a constant.
     *
     * @param value: The value to be assigned.
     * @return TensorCondProxy: The tensor condition proxy object.
     */
    TensorCondProxy& operator=(const T value) {
        thrust::transform_if(this->begin(), this->end(), cond.begin(), this->begin(), AssignValue<T>(value), [] __device__(const bool& x) { return x; });
        return *this;
    }

   private:
    Tensor<bool, N> cond;
};

}  // namespace vt
