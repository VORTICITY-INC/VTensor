#pragma once

#include <array>

#include "lib/core/iterator.hpp"
#include "lib/core/tensor.hpp"

namespace vt {

/**
 * @brief CuTensor class. It is used to adapt a Tensor to be used in CUDA kernel.
 *
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 */
template <typename T, size_t N>
class CuTensor {
   public:
    /**
     * @brief Construct a new CuTensor object from a Tensor.
     *
     * @param tensor: The tensor to be adapted.
     */
    __host__ CuTensor(Tensor<T, N> tensor)
        : CuTensor(tensor.raw_ptr(), tensor.shape().data(), tensor.strides().data(), tensor.start(), tensor.size(), tensor.order(), tensor.contiguous()) {}

    /**
     * @brief Construct a new CuTensor object from a raw pointer, shape, strides, start index, and contiguous flag.
     *
     * @param data: The raw pointer to the data.
     * @param shape: The shape of the tensor.
     * @param strides: The strides of the tensor.
     * @param start: The start index of the tensor.
     * @param size: The size of the tensor.
     * @param order: The order of the tensor.
     * @param contiguous: The contiguous flag of the tensor.
     */
    __host__ __device__ CuTensor(T* data, const size_t* shape, const size_t* strides, const size_t start, const size_t size, const Order order,
                                 const bool contiguous)
        : data(data), start(start), size(size), order(order), contiguous(contiguous) {
        if constexpr (N > 0) {
            for (size_t i = 0; i < N; ++i) {
                this->shape[i] = shape[i];
                this->strides[i] = strides[i];
            }
        }
    }

    /**
     * @brief Helper function to get the tensor index. (e.x. tensor(0, 1, 2))
     *
     * @param args: The indices of the tensor.
     * @tparam Args: Variadic template for mulitple arguments.
     * @return size_t: The index of the tensor.
     */
    template <typename... Args>
    __host__ __device__ size_t get_tensor_index(Args... args) const {
        static_assert(sizeof...(args) == N, "Number of indices must match tensor dimensions");
        size_t index = start;
        if constexpr (N > 0) {
            size_t indices[] = {static_cast<size_t>(args)...};
            for (size_t i = 0; i < N; ++i) index += indices[i] * strides[i];
        }
        return index;
    }

    /**
     * @brief Return the data given the indices.
     *
     * @param args: The indices of the tensor.
     * @tparam Args: Variadic template for mulitple arguments.
     * @return T: The data of the tensor.
     */
    template <typename... Args>
    __host__ __device__ T& operator()(Args... args) {
        return data[get_tensor_index(args...)];
    }

    /**
     * @brief Return the data given the indices.
     *
     * @param args: The indices of the tensor.
     * @tparam Args: Variadic template for mulitple arguments.
     * @return T: The data of the tensor.
     */
    template <typename... Args>
    __host__ __device__ const T& operator()(Args... args) const {
        return data[get_tensor_index(args...)];
    }

    /**
     * @brief Return the data given the iterative index
     *
     * @param n: The 1D index of the tensor.
     * @return T: The data of the tensor.
     */
    __host__ __device__ T& operator[](size_t n) { return contiguous ? data[n] : data[get_iterator_index<N>(n, shape, strides, start, order)]; }

    /**
     * @brief Return the data given the iterative index
     *
     * @param n: The 1D index of the tensor.
     * @return T: The data of the tensor.
     */
    __host__ __device__ const T& operator[](size_t n) const { return contiguous ? data[n] : data[get_iterator_index<N>(n, shape, strides, start, order)]; }

    size_t shape[N];
    size_t strides[N];
    size_t start;
    size_t size = 0;
    bool contiguous;
    Order order;
    T* data;
};

}  // namespace vt
