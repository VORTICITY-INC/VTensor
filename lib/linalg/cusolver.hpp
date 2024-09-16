#pragma once

#include <cublas_v2.h>
#include <cusolverDn.h>

namespace vt {

namespace cuda {

/**
 * @brief check the status of the CuSolver library.
 *
 * @param status: The status of the CuSolver library.
 * @param message: The message to be displayed if the status is not CUSOLVER_STATUS_SUCCESS.
 * @throw std::runtime_error: If the status is not CUSOLVER_STATUS_SUCCESS.
 */
inline void check_cusolver_status(cusolverStatus_t status, const std::string& message) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        throw std::runtime_error(message + ": " + std::to_string(status));
    }
}

/**
 * @brief Deleter for the CuSolver handle.
 */
struct CuSolverHandleDeleter {
    void operator()(cusolverDnHandle_t* handle) const {
        if (handle) {
            check_cusolver_status(cusolverDnDestroy(*handle), "Failed to destroy CuSolver handle");
            delete handle;
        }
    }
};

// Alias for the CuSolver unique pointer handle
using CuSolverHandleT = std::unique_ptr<cusolverDnHandle_t, CuSolverHandleDeleter>;

/**
 * @brief Create a CuSolver unique pointer handle.
 *
 * @return CuSolverHandleT: The CuSolver unique pointer handle.
 */
inline CuSolverHandleT create_cusolver_handle() {
    auto handle = CuSolverHandleT(new cusolverDnHandle_t);
    check_cusolver_status(cusolverDnCreate(handle.get()), "Failed to create CuSolver handle");
    return handle;
}

// Singleton class for global CuSolver handle management
class CuSolver {
   public:
    // Delete copy constructor and assignment operator to ensure Singleton
    CuSolver(const CuSolver&) = delete;
    CuSolver& operator=(const CuSolver&) = delete;

    /**
     * @brief Get the Instance object. It returns the singleton instance of the CuSolver handle.
     *
     * @return CuSolver&
     */
    static CuSolver& get_instance() {
        static CuSolver instance;
        return instance;
    }

    /**
     * @brief Get the CuSolver handle.
     *
     * @return cusolverDnHandle_t: The CuSolver handle.
     */
    cusolverDnHandle_t get_handle() const { return *handle.get(); }

   private:
    // Private constructor to enforce Singleton
    CuSolver() { handle = create_cusolver_handle(); }

    // Unique pointer to manage the CuSolver handle
    CuSolverHandleT handle;
};

// This is global CuSolver instance
static CuSolver& cusolver = CuSolver::get_instance();

/**
 * @brief The type of the CuSolver functions.
 *
 * @tparam T: The data type of the CuSolver functions.
 */
template <typename T>
struct CuSolverFuncType {
    using GetRFT = cusolverStatus_t (*)(cusolverDnHandle_t, int, int, T*, int, T*, int*, int*);
    using GetRST = cusolverStatus_t (*)(cusolverDnHandle_t, cublasOperation_t, int, int, const T*, int, const int*, T*, int, int*);
    using GetRFBufferSizeT = cusolverStatus_t (*)(cusolverDnHandle_t, int, int, T*, int, int*);
};

/**
 * @brief The CuSolver functions for different data types.
 *
 * @tparam T: The data type of the CuSolver functions.
 */
template <typename T>
struct CuSolverFunc {};

/**
 * @brief The CuSolver functions for float data type.
 */
template <>
struct CuSolverFunc<float> {
    static constexpr CuSolverFuncType<float>::GetRFT getrf() { return cusolverDnSgetrf; }
    static constexpr CuSolverFuncType<float>::GetRST getrs() { return cusolverDnSgetrs; }
    static constexpr CuSolverFuncType<float>::GetRFBufferSizeT getrf_buffer_size() { return cusolverDnSgetrf_bufferSize; }
};

/**
 * @brief The CuSolver functions for double data type.
 */
template <>
struct CuSolverFunc<double> {
    static constexpr CuSolverFuncType<double>::GetRFT getrf() { return cusolverDnDgetrf; }
    static constexpr CuSolverFuncType<double>::GetRST getrs() { return cusolverDnDgetrs; }
    static constexpr CuSolverFuncType<double>::GetRFBufferSizeT getrf_buffer_size() { return cusolverDnDgetrf_bufferSize; }
};

}  // namespace cuda
}  // namespace vt
