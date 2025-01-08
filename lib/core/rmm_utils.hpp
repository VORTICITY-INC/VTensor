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
    Mempool(unsigned long pool_size, const std::string& log_filepath);

   private:
    class MempoolImpl;
    std::unique_ptr<MempoolImpl> pimpl;
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
