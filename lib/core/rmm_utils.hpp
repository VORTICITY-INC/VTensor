#pragma once

#include <cuda_runtime.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace vt {

/**
 * @brief Check the CUDA errors.
 *
 * @param err: The CUDA error code.
 * @param message: The error message.
 */
void check_cuda_errors(cudaError_t err, const std::string& message);

/**
 * @brief Get the number of GPUs.
 *
 * @return int: The number of GPUs.
 */
int get_number_of_gpus();

/**
 * @brief A class that manages a memory pool for the current GPU.
 */
class Mempool {
   public:
    /**
     * @brief Construct a Mempool object.
     *
     * @param pool_size: The size of the memory pool.
     * @param log_filepath: The log file path for memory usage logging.
     */
    Mempool(size_t pool_size, const std::string& log_filepath);

    /**
     * @brief Destroy the Mempool object
     *
     */
    ~Mempool();

   private:
    class MempoolImpl;
    std::unique_ptr<MempoolImpl> pimpl;
};

/**
 * @brief A class that manages a pinned memory pool.
 *
 */
class PinnedMempool {
   public:
    /**
     * @brief Construct a PinnedMempool object.
     *
     * @param initial_pinned_pool_size: The initial size of the pinned memory pool.
     * @param pinned_pool_size: The size of the pinned memory pool.
     */
    PinnedMempool(size_t initial_pinned_pool_size, size_t pinned_pool_size);

    /**
     * @brief Destroy the PinnedMempool object
     *
     */
    ~PinnedMempool();

    /**
     * @brief Deallocate the memory for the given pointer and size.
     *
     * @param ptr: The pointer to the memory.
     * @param size: The size of the memory.
     */
    void deallocate(void* ptr, size_t size);

    /**
     * @brief Allocate the memory for the given size.
     *
     * @param size: The size of the memory.
     * @return void*: The pointer to the allocated memory.
     */
    void* allocate(size_t size);

   private:
    class PinnedMempoolImpl;
    std::unique_ptr<PinnedMempoolImpl> pimpl;
};

/**
 * @brief Singleton class for global memory pool management
 *
 */
class GlobalMempool {
   public:
    // Delete copy constructor and assignment operator to ensure Singleton
    GlobalMempool(const GlobalMempool&) = delete;
    GlobalMempool& operator=(const GlobalMempool&) = delete;

    /**
     * @brief Get the Instance object. It returns the singleton instance of the global memory pool.
     *
     * @param pool_size: The size of the memory pool.
     * @return GlobalMempool&: The global memory pool instance.
     */
    static GlobalMempool& get_instance(size_t pool_size);

   private:
    /**
     * @brief Construct a new GlobalMempool object.
     *
     * @param pool_size: The size of the memory pool.
     */
    GlobalMempool(size_t pool_size);

    std::vector<std::unique_ptr<Mempool>> mempools;
};

}  // namespace vt
