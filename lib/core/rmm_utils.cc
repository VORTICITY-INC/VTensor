#include "lib/core/rmm_utils.hpp"
#include <rmm/device_buffer.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/logging_resource_adaptor.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>


namespace vt {

    void check_cuda_errors(cudaError_t err, const std::string& message) {
        if (err != cudaSuccess) throw std::runtime_error(message + ": " + std::to_string(err));
    }

    int get_number_of_gpus() {
        int device_count = 0;
        check_cuda_errors(cudaGetDeviceCount(&device_count), "Failed to get the number of GPUs");
        return device_count;
    }

    /**
     * @brief Private implementation pointer. This could reduce compilation time for RMM headers. The implementation is defined in rmm_utils.cc.
     *
     */
    class Mempool::MempoolImpl {
        public:
            /**
             * @brief Construct a new Mempool Impl object
             * 
             * @param pool_size: The size of the memory pool.
             * @param log_filepath: The log file path for memory usage logging.
             */
            MempoolImpl(size_t pool_size, const std::string& log_filepath)
                : log_mr{&cuda_mr, log_filepath}, pool_mr(&log_mr, rmm::percent_of_free_device_memory(pool_size)) {
                rmm::mr::set_current_device_resource(&pool_mr);
            }

        private:
            rmm::mr::cuda_memory_resource cuda_mr;
            rmm::mr::logging_resource_adaptor<rmm::mr::cuda_memory_resource> log_mr;
            rmm::mr::pool_memory_resource<rmm::mr::logging_resource_adaptor<rmm::mr::cuda_memory_resource>> pool_mr;
    };

    Mempool::Mempool(unsigned long pool_size, const std::string& log_filepath)
        : pimpl(std::make_unique<MempoolImpl>(pool_size, log_filepath)) {}
        
    GlobalMempool& GlobalMempool::get_instance(size_t pool_size) {
        static GlobalMempool instance(pool_size);
        return instance;
    }

    GlobalMempool::GlobalMempool(size_t pool_size) {
        for (auto i = 0; i < get_number_of_gpus(); ++i) {
            check_cuda_errors(cudaSetDevice(i), "Failed to set device");
            auto log_filepath = "/tmp/vtensor/memory" + std::to_string(i) + ".log";
            mempools.emplace_back(std::make_unique<Mempool>(pool_size, log_filepath));
        }
    }

} // namespace vt