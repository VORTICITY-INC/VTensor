#pragma once

#include <cublas_v2.h>

namespace vt {

namespace cuda {

/**
 * @brief check the status of the CUBLAS library.
 *
 * @param status: The status of the CUBLAS library.
 * @param message: The message to be displayed if the status is not CUBLAS_STATUS_SUCCESS.
 * @throw std::runtime_error: If the status is not CUBLAS_STATUS_SUCCESS.
 */
inline void check_cublas_status(cublasStatus_t status, const std::string& message) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(message + ": " + std::to_string(status));
    }
}

/**
 * @brief Deleter for the CuBLAS handle.
 */
struct CuBLASHandleDeleter {
    void operator()(cublasHandle_t* handle) const {
        if (handle) {
            check_cublas_status(cublasDestroy(*handle), "Failed to destroy CuBLAS handle");
            delete handle;
        }
    }
};

// Alias for the CuBLAS unique pointer handle
using CuBLASHandleT = std::unique_ptr<cublasHandle_t, CuBLASHandleDeleter>;

/**
 * @brief Create a CuBLAS unique pointer handle.
 *
 * @return CuBLASHandleT: The CuBLAS unique pointer handle.
 */
inline CuBLASHandleT create_cublas_handle() {
    auto handle = CuBLASHandleT(new cublasHandle_t);
    check_cublas_status(cublasCreate(handle.get()), "Failed to create CuBLAS handle");
    return handle;
}

// Singleton class for global CuBLAS handle management
class CuBLAS {
   public:
    // Delete copy constructor and assignment operator to ensure Singleton
    CuBLAS(const CuBLAS&) = delete;
    CuBLAS& operator=(const CuBLAS&) = delete;

    /**
     * @brief Get the Instance object. It returns the singleton instance of the CuBLAS handle.
     *
     * @return CuBLAS&
     */
    static CuBLAS& get_instance() {
        static CuBLAS instance;
        return instance;
    }

    /**
     * @brief Get the CuBLAS handle.
     *
     * @return cublasHandle_t: The CuBLAS handle.
     */
    cublasHandle_t get_handle() const { return *handle.get(); }

   private:
    // Private constructor to enforce Singleton
    CuBLAS() { handle = create_cublas_handle(); }

    // Unique pointer to manage the CuBLAS handle.
    CuBLASHandleT handle;
};

/**
 * @brief The type of the CuBLAS functions.
 *
 * @tparam T: The data type of the CuBLAS functions.
 */
template <typename T>
struct CuBLASFuncType {
    using DotT = cublasStatus_t (*)(cublasHandle_t, int, const T*, int, const T*, int, T*);
    using GeamT = cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, const T*, const T*, int, const T*, const T*, int, T*, int);
    using GemmT = cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const T*, const T*, int, const T*, int, const T*, T*,
                                     int);
    using GemmBatchedT = cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const T*, const T*, int, long long int,
                                            const T*, int, long long int, const T*, T*, int, long long int, int);
    using GetRFBatchedT = cublasStatus_t (*)(cublasHandle_t, int, T* const[], int, int*, int*, int);
    using GetRIBatchedT = cublasStatus_t (*)(cublasHandle_t, int, const T* const[], int, const int*, T* const[], int, int*, int);
};

/**
 * @brief The CuBLAS functions for different data types.
 *
 * @tparam T: The data type of the CuBLAS functions.
 */
template <typename T>
struct CuBLASFunc {};

/**
 * @brief The CuBLAS functions for float data type.
 */
template <>
struct CuBLASFunc<float> {
    static constexpr CuBLASFuncType<float>::DotT dot() { return cublasSdot; }
    static constexpr CuBLASFuncType<float>::GeamT geam() { return cublasSgeam; }
    static constexpr CuBLASFuncType<float>::GemmT gemm() { return cublasSgemm; }
    static constexpr CuBLASFuncType<float>::GemmBatchedT gemm_batched() { return cublasSgemmStridedBatched; }
    static constexpr CuBLASFuncType<float>::GetRFBatchedT getrf_batched() { return cublasSgetrfBatched; }
    static constexpr CuBLASFuncType<float>::GetRIBatchedT getri_batched() { return cublasSgetriBatched; }
};

/**
 * @brief The CuBLAS functions for double data type.
 */
template <>
struct CuBLASFunc<double> {
    static constexpr CuBLASFuncType<double>::DotT dot() { return cublasDdot; }
    static constexpr CuBLASFuncType<double>::GeamT geam() { return cublasDgeam; }
    static constexpr CuBLASFuncType<double>::GemmT gemm() { return cublasDgemm; }
    static constexpr CuBLASFuncType<double>::GemmBatchedT gemm_batched() { return cublasDgemmStridedBatched; }
    static constexpr CuBLASFuncType<double>::GetRFBatchedT getrf_batched() { return cublasDgetrfBatched; }
    static constexpr CuBLASFuncType<double>::GetRIBatchedT getri_batched() { return cublasDgetriBatched; }
};

}  // namespace cuda
}  // namespace vt
