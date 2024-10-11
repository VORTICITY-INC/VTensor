#pragma once

#ifndef POOL_SIZE
#define POOL_SIZE 50
#endif

#ifndef DISABLE_GLOBAL_MEMPOOL
#define DISABLE_GLOBAL_MEMPOOL false
#endif

#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/logging_resource_adaptor.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

namespace vt {

/**
 * @brief Check the CUDA errors.
 *
 * @param err: The CUDA error code.
 * @param message: The error message.
 */
inline void check_cuda_errors(cudaError_t err, const std::string& message) {
    if (err != cudaSuccess) throw std::runtime_error(message + ": " + std::to_string(err));
}

/**
 * @brief Get the number of GPUs.
 *
 * @return int: The number of GPUs.
 */
inline int get_number_of_gpus() {
    int device_count = 0;
    check_cuda_errors(cudaGetDeviceCount(&device_count), "Failed to get the number of GPUs");
    return device_count;
}

// Setup a memory pool for the current GPU
class Mempool {
   public:
    /**
     * @brief Construct a Mempool object.
     *
     * @param pool_size: The size of the memory pool.
     * @param log_filepath: The log file path.
     */
    Mempool(size_t pool_size, const std::string& log_filepath = "/tmp/vtensor/memory.log")
        : log_mr{&cuda_mr, log_filepath}, pool_mr(&log_mr, rmm::percent_of_free_device_memory(pool_size)) {
        rmm::mr::set_current_device_resource(&pool_mr);
    }

   private:
    rmm::mr::cuda_memory_resource cuda_mr;
    rmm::mr::logging_resource_adaptor<rmm::mr::cuda_memory_resource> log_mr;
    rmm::mr::pool_memory_resource<rmm::mr::logging_resource_adaptor<rmm::mr::cuda_memory_resource>> pool_mr;
};

// Singleton class for global memory pool management
class GlobalMempool {
   public:
    // Delete copy constructor and assignment operator to ensure Singleton
    GlobalMempool(const GlobalMempool&) = delete;
    GlobalMempool& operator=(const GlobalMempool&) = delete;

    /**
     * @brief Get the Instance object. It returns the singleton instance of the global memory pool.
     *
     * @return GlobalMempool&: The global memory pool instance.
     */
    static GlobalMempool& get_instance() {
        static GlobalMempool instance;
        return instance;
    }

   private:
    /**
     * @brief Construct a new GlobalMempool object.
     */
    GlobalMempool() {
        size_t pool_size = POOL_SIZE;
        for (auto i = 0; i < get_number_of_gpus(); ++i) {
            check_cuda_errors(cudaSetDevice(i), "Failed to set device");
            auto log_filepath = "/tmp/vtensor/memory" + std::to_string(i) + ".log";
            mempools.emplace_back(std::make_unique<Mempool>(pool_size, log_filepath));
        }
    }

    std::vector<std::unique_ptr<Mempool>> mempools;
};

class GlobalMempoolInitializer {
   public:
    GlobalMempoolInitializer() {
        if (!DISABLE_GLOBAL_MEMPOOL) {
            GlobalMempool::get_instance();
        }
    }
};

static GlobalMempoolInitializer mempool_initializer;

}  // namespace vt
