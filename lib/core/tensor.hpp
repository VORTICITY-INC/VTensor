#pragma once

#include <thrust/iterator/constant_iterator.h>

#include <memory>
#include <numeric>
#include <rmm/device_vector.hpp>
#include <lib/core/iterator.hpp>
#include <lib/core/slice.hpp>

namespace vt {

// Number of threads per block. It is used for the customized CUDA kernel.
const size_t NUM_THREADS_X = 256;

// Alias for shape of the tensor.
template <size_t N>
using Shape = std::array<size_t, N>;

/**
 * @brief Helper function to calculate the size of a tensor.
 *
 * @param shape: Shape of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @return size: Size of the tensor.
 */
template <size_t N>
size_t get_size(const Shape<N>& shape) {
    return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
}

/**
 * @brief Helper function to calculate the strides of a tensor.
 *
 * @param shape: Shape of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * @return strides: Strides of the tensor.
 */
template <size_t N>
Shape<N> get_strides(const Shape<N>& shape) {
    Shape<N> strides{};
    size_t stride = 1;
    for (int i = N - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }
    return strides;
}

/**
 * @brief Tensor class. It is used to represent a tensor from thrust device vector.
 * The concept of the tesnor is similar to the numpy array in Python.
 * It stored a shared pointer to the data, and shape, strides, start index of the tensor.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 */
template <typename T, size_t N>
class Tensor {
   public:
    using vector_type = rmm::device_vector<T>;

    /**
     * @brief Construct a new tensor object from a shape.
     *
     * @param shape Shape of the tensor.
     */
    Tensor(const Shape<N>& shape) : _shape(shape), _size(get_size(shape)), _strides(get_strides(shape)), _start(0), _sliced(false) {
        _data = std::make_shared<vector_type>(_size);
    }

    /**
     * @brief Construct a new tensor object from a shared vector and a shape.
     *
     * @param data A shared vector
     * @param shape Shape of the tensor.
     */
    Tensor(const std::shared_ptr<vector_type>& data, const Shape<N>& shape)
        : _data(data), _shape(shape), _size(get_size(shape)), _strides(get_strides(shape)), _start(0), _sliced(false) {}

    /**
     * @brief Construct a new tensor object from a shared vector and a shape, strides, start index.
     *
     * @param data A shared vector
     * @param shape Shape of the tensor.
     * @param strides Strides of the tensor.
     * @param start Start index of the tensor.
     * @param sliced Sliced flag of the tensor.
     */
    Tensor(const std::shared_ptr<vector_type>& data, const Shape<N>& shape, const Shape<N>& strides, size_t start, bool sliced = false)
        : _data(data), _shape(shape), _size(get_size(shape)), _strides(strides), _start(start), _sliced(sliced) {}

    /**
     * @brief Reshape the array to a new shape.
     * If the tensor has been sliced, the method will copy the tensor and reshape the new tensor.
     *
     * @tparam Args: Variadic template for mulitple arguments.
     * @param args: New shape for the tensor.
     * @return Tensor: The reshaped tensor.
     */
    template <typename... Args>
    Tensor<T, sizeof...(Args)> reshape(Args... args) const {
        auto constexpr dim = sizeof...(args);
        std::array<size_t, dim> shape = {static_cast<size_t>(args)...};
        assert(_size == get_size(shape));
        auto tensor = ascontiguoustensor(*this);
        return Tensor<T, dim>(tensor.data(), shape);
    }

    /**
     * @brief Operator for indexing the tensor along the last axis (row-major order).
     * Notice that the method doesn't gauranttee the contiguity of the tensor. If the tensor has been transposed, the return slice would not be contiguous.
     *
     * @param index: The index of the tensor.
     * @return Tensor: The sliced tensor.
     */
    Tensor<T, N - 1> operator[](size_t index) const {
        static_assert(N > 1, "The tensor must be at least 2D.");
        Shape<N - 1> new_shape;
        Shape<N - 1> new_strides;
        std::copy(_shape.begin() + 1, _shape.end(), new_shape.begin());
        std::copy(_strides.begin() + 1, _strides.end(), new_strides.begin());
        auto new_start = _start + index * _strides[0];
        return Tensor<T, N - 1>(_data, new_shape, new_strides, new_start, true);
    }

    /**
     * @brief Operator for slicing the tensor. It could receive multiple slices through explicit Slice constructor.
     *
     * @param args: Slices to be applied to the tensor.
     * @tparam Args: Variadic template for mulitple arguments.
     * @return TensorSliceProxy: The tensor slice proxy object.
     */
    template <typename... Args>
    TensorSliceProxy<T, N> operator()(Args... args) {
        std::array<Slice, N> slices = {static_cast<Slice>(args)...};
        return TensorSliceProxy<T, N>(apply_slices(slices));
    }

    /**
     * @brief Operator for slicing the tensor. It supports one Slice implicit conversion for tensor.
     *
     * @param slice: Slice to be applied to the tensor.
     * @return TensorSliceProxy: The tensor slice proxy object.
     */
    TensorSliceProxy<T, 1> operator()(Slice s) {
        auto slices = std::array<Slice, 1>{s};
        return TensorSliceProxy<T, 1>(apply_slices(slices));
    }

    /**
     * @brief Operator for slicing the tensor. It supports two Slice implicit conversion for tensor.
     *
     * @param slice1: Slice to be applied to the tensor.
     * @param slice2: Slice to be applied to the tensor.
     * @return TensorSliceProxy: The tensor slice proxy object.
     */
    TensorSliceProxy<T, 2> operator()(Slice s1, Slice s2) {
        auto slices = std::array<Slice, 2>{s1, s2};
        return TensorSliceProxy<T, 2>(apply_slices(slices));
    }

    /**
     * @brief Operator for slicing the tensor. It supports three Slice implicit conversion for tensor.
     *
     * @param slice1: Slice to be applied to the tensor.
     * @param slice2: Slice to be applied to the tensor.
     * @param slice3: Slice to be applied to the tensor.
     * @return TensorSliceProxy: The tensor slice proxy object.
     */
    TensorSliceProxy<T, 3> operator()(Slice s1, Slice s2, Slice s3) {
        auto slices = std::array<Slice, 3>{s1, s2, s3};
        return TensorSliceProxy<T, 3>(apply_slices(slices));
    }

    /**
     * @brief Operator for in-place addition of a vector and a scalar.
     *
     * @param value: The scalar value to be added.
     * @return Tensor: The left-hand side tensor object.
     */
    Tensor operator+=(const T value) const {
        thrust::transform(this->begin(), this->end(), thrust::make_constant_iterator(value), this->begin(), thrust::plus<T>());
        return *this;
    }

    /**
     * @brief Operator for in-place minus of a vector and a scalar.
     *
     * @param value: The scalar value to be minus.
     * @return Tensor: The left-hand side tensor object.
     */
    Tensor operator-=(const T value) const {
        thrust::transform(this->begin(), this->end(), thrust::make_constant_iterator(value), this->begin(), thrust::minus<T>());
        return *this;
    }

    /**
     * @brief Operator for in-place addition of two vectors.
     *
     * @param other: The other tensor to be added.
     * @return Tensor: The left-hand side tensor object.
     */
    Tensor operator+=(const Tensor& other) const {
        assert(_shape == other.shape());
        thrust::transform(this->begin(), this->end(), other.begin(), this->begin(), thrust::plus<T>());
        return *this;
    }

    /**
     * @brief Operator for in-place minus of two vectors.
     *
     * @param other: The other tensor to be minus.
     * @return Tensor: The left-hand side tensor object.
     */
    Tensor operator-=(const Tensor& other) const {
        assert(_shape == other.shape());
        thrust::transform(this->begin(), this->end(), other.begin(), this->begin(), thrust::minus<T>());
        return *this;
    }

    /**
     * @brief Return the begin iterator for the Tensor.
     *
     * @return TensorIterator: The iterator points to the 1st index of the tensor.
     */
    TensorIterator<typename vector_type::iterator, N> begin() const {
        return TensorIterator<typename vector_type::iterator, N>(_data->begin(), _shape.data(), _strides.data(), _start, _sliced);
    }

    /**
     * @brief Return the end iterator for the Tensor.
     *
     * @return TensorIterator: The iterator points to the last index of the tensor.
     */
    TensorIterator<typename vector_type::iterator, N> end() const {
        return TensorIterator<typename vector_type::iterator, N>(_data->begin() + _size, _shape.data(), _strides.data(), _start, _sliced);
    }

    /**
     * @brief Apply slices the tensor and return the new tensor object.
     *
     * @param slices: The slices to be applied to the tensor.
     * @return Tensor: The new tensor object.
     */
    Tensor apply_slices(const std::array<Slice, N>& slices) {
        Shape<N> new_strides{};
        Shape<N> new_shape{};
        size_t offset = 0;
        for (size_t i = 0; i < N; i++) {
            assert(slices[i].end <= _shape[i]);       // Check if the end of the slice is within the tensor dimension.
            assert(slices[i].start < slices[i].end);  // Check if the start of the slice is less than the end.
            new_strides[i] = slices[i].step * _strides[i];
            new_shape[i] = (slices[i].end - slices[i].start + slices[i].step - 1) / slices[i].step;
            offset += slices[i].start * _strides[i];
        }
        size_t new_start = _start + offset;
        return Tensor<T, N>(_data, new_shape, new_strides, new_start, true);
    }

    /**
     * @brief Return the raw pointer of the tensor.
     *
     * @return T*: The raw pointer of the tensor.
     */
    T* raw_ptr() { return thrust::raw_pointer_cast(_data->data()); }

    /**
     * @brief Return the sliced flag of the tensor.
     *
     * @return bool: The sliced flag of the tensor.
     */
    bool sliced() const { return _sliced; }

    /**
     * @brief Return the size of the tensor.
     *
     * @return size_t: The size of the tensor.
     */
    size_t size() const { return _size; }

    /**
     * @brief Return the start index of the tensor.
     *
     * @return size_t: The start index of the tensor.
     */
    size_t start() const { return _start; }

    /**
     * @brief Return the shape of the tensor.
     *
     * @return Shape: The shape of the tensor.
     */
    Shape<N> shape() const { return _shape; }

    /**
     * @brief Return the strides of the tensor.
     *
     * @return Shape: The strides of the tensor.
     */
    Shape<N> strides() const { return _strides; }

    /**
     * @brief Return the shared pointer of the data.
     *
     * @return std::shared_ptr<vector_type>: The shared pointer of the data.
     */
    std::shared_ptr<vector_type> data() const { return _data; }

    /**
     * @brief Set the strides of the tensor.
     *
     * @param strides: The strides of the tensor.
     */
    void set_strides(Shape<N> strides) { _strides = strides; }

   private:
    size_t _start = 0;
    size_t _size = 0;
    Shape<N> _shape;
    Shape<N> _strides;
    bool _sliced = false;
    std::shared_ptr<vector_type> _data = nullptr;
};

}  // namespace vt
