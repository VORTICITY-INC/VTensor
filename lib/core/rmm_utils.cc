#include "lib/core/rmm_utils.hpp"
#include <rmm/device_buffer.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/logging_resource_adaptor.hpp>
#include "rmm/mr/device/pool_memory_resource.hpp"
#include "rmm/mr/host/pinned_memory_resource.hpp"

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
     * @brief Private implementation pointer. This could reduce compilation time for RMM headers.
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
    
    Mempool::~Mempool() = default;

    Mempool::Mempool(size_t pool_size, const std::string& log_filepath)
        : pimpl(std::make_unique<MempoolImpl>(pool_size, log_filepath)) {}
        

    /**
     * @brief Private implementation pointer for PinnedMempool. This could reduce compilation time for RMM headers.
     * 
     */
    class PinnedMempool::PinnedMempoolImpl {
        public:
            /**
             * @brief Construct a new Pinned Mempool Impl object
             * 
             * @param initial_pinned_pool_size: The initial size of the pinned memory pool.
             * @param pinned_pool_size: The size of the pinned memory pool.
             */
            PinnedMempoolImpl(size_t initial_pinned_pool_size, size_t pinned_pool_size)
                : pool_mr(pinned_mr, initial_pinned_pool_size, pinned_pool_size) {}

            rmm::mr::pinned_memory_resource pinned_mr;
            rmm::mr::pool_memory_resource<rmm::mr::pinned_memory_resource> pool_mr;
    };

    PinnedMempool::~PinnedMempool() = default;

    void PinnedMempool::deallocate(void* ptr, size_t size) {
        pimpl->pool_mr.deallocate(ptr, size);
    }

    void* PinnedMempool::allocate(size_t size) {
        return pimpl->pool_mr.allocate(size);
    }

    PinnedMempool::PinnedMempool(size_t initial_pinned_pool_size, size_t pinned_pool_size)
        : pimpl(std::make_unique<PinnedMempoolImpl>(initial_pinned_pool_size, pinned_pool_size)) {}

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