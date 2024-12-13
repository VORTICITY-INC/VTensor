
#include "BS_thread_pool.hpp" 
#include <rmm/mr/host/pinned_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <iostream>
#include <lib/vtensor.hpp>
#include <variant>

class CPUTimer {
   public:
    /**
     * @brief Construct a new CPUTimer object.
     */
    CPUTimer() : start_time_point(), end_time_point(), running(false) {}

    /**
     * @brief Start the timer.
     */
    void start() {
        start_time_point = std::chrono::high_resolution_clock::now();
    }

    /**
     * @brief Stop the timer.
     */
    double stop() {
        end_time_point = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end_time_point - start_time_point).count();
    }

   private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time_point;
    std::chrono::time_point<std::chrono::high_resolution_clock> end_time_point;
    bool running;
};

/**
 * @brief Buffer class for cache.
 * 
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 */
template <typename T, size_t N>
struct Buffer {
    T* data;
    size_t size;
    std::string filepath;
    vt::Shape<N> shape;
};

/**
 * @brief IO handler for saving and loading tensors asynchronously.
 * 
 * @tparam T: Data type of the tensor.
 * @tparam N: Number of dimensions of the tensor.
 * 
 * [TODO] unordered_map is not thread-safe. We could use a concurrent hash map in the future.
 * [TODO] Need to check if column-major memory layout works.
 * ------------------------------------------
 * @section case1 Case 1: Save fields to disk for movie, no cache, saved to disk via pinned memory
 * 
 * @code
 * // Create an IOHandler object
 * // total_pool_size: Matched with the total number of data size in the tensor maps.
 * auto io = IOHandler<float, 1>(total_pool_size, total_pool_size, num_workers);
 * 
 * // Create a tensor map for the fields you want to save
 * IOHandler<float, 1>::tensor_map_type tensors;
 * tensors["/mammoth/movie/0.npy"] = fields;
 * 
 * // Save to disk asynchronously
 * io.save(tensors);
 * 
 * // Call sync to wait for all the tasks to finish before saving the next set of fields
 * io.sync();
 * @endcode
 * 
 * ------------------------------------------
 * @section case2 Case 2: Checkpoint, fully cache pinned memory
 * 
 * @code
 * // Create a tensor map for the checkpoint fields
 * IOHandler<float, 1>::tensor_map_type tensors;
 * tensors["/mammoth/checkpoints/0.npy"] = fields;
 * 
 * // Create an IOHandler object
 * // num_files: Matched with the number of elements in the tensor maps.
 * // initial_pool_size: Could be any number from 0 to total_pool_size, but it is better to match with total_pool_size.
 * // total_pool_size: Total number of bytes for all the checkpoints.
 * auto io = IOHandler<float, 1>(initial_pool_size, total_pool_size, num_workers);
 * 
 * // Put pinned memory in cache
 * io.save(tensors, true);
 * 
 * // Call sync to wait for all the tasks to finish before saving the next set of fields
 * io.sync();
 * 
 * // Load the fields from the cache
 * IOHandler<float, 1>::array_map_type arrays;
 * arrays["/mammoth/checkpoints/0.npy"] = std::monostate{};
 * 
 * // Asynchronously load the fields from the cache
 * io.load_from_disk_to_host(arrays);
 * 
 * // Call sync to wait for all the tasks to finish before transferring the fields to the device
 * io.sync();
 * auto tensors = io.load_from_host_to_device(arrays);
 * @endcode
 * 
 * ------------------------------------------
 * @section case3 Case 3: Checkpoint, partially cache pinned memory
 * 
 * @code
 * // Create a tensor map for the checkpoint fields
 * IOHandler<float, 1>::tensor_map_type tensors;
 * tensors["/mammoth/checkpoints/0.npy"] = fields;
 * 
 * // Create an IOHandler object
 * // num_files: Matched with the number of elements in the tensor maps.
 * // initial_pool_size: Could be any number from 0 to total_pool_size, but it is better to match with total_pool_size.
 * // total_pool_size: Total number of bytes for the checkpoints.
 * auto io = IOHandler<float, 1>(initial_pool_size, total_pool_size, num_workers);
 * 
 * // Put pinned memory in cache
 * io.save(tensors, true);
 * 
 * // Call sync to wait for all the tasks to finish before saving the next set of fields
 * io.sync();
 * 
 * // Save to disk asynchronously if no pinned memory available in the pool.
 * // Currently, users need to decide whether to save to disk or not. In the future, we could add a mechanism to automatically save to disk when the pinned memory pool is full.
 * io.save(tensors);
 * 
 * // Call sync to wait for all the tasks to finish before saving the next set of fields
 * io.sync();
 * 
 * // The loading part is the same as Case 2.
 * @endcode
 */
template <typename T, size_t N>
class IOHandler {

    public:

    using array_type = xt::xarray<T>;
    using tensor_type = vt::Tensor<T, N>;
    using buffer_type = Buffer<T, N>;
    using array_variant = std::variant<std::monostate, std::shared_ptr<array_type>, buffer_type>;
    using array_map_type = std::unordered_map<std::string, array_variant>;
    using tensor_map_type = std::unordered_map<std::string, tensor_type>;

    /**
     * @brief Construct a new IOHandler object
     * 
     * @param initial_pinned_pool_size: Initial size of the pinned memory pool. The CudaHostAllocator will allocate this amount of pinned memory at the beginning.
     * @param pinned_pool_size: Maximum size of the pinned memory pool.
     * @param num_threads: Number of threads for the thread pool for asynchronize saving and loading.
     */
    IOHandler(size_t initial_pinned_pool_size, size_t pinned_pool_size, int num_threads) : thread_pool(num_threads), pool_mr(pinned_mr, initial_pinned_pool_size, pinned_pool_size) {}

    /**
     * @brief Destroy the IOHandler object
     * 
     *//
    ~IOHandler() { flush(); }

    /**
     * @brief Flush the caches, deallocate pinned memory and reset thread pool.
     * 
     */
    void flush() { 
        // Reset thread pool
        thread_pool.reset();

        // Clean up caches and deallocate pinned memory
        for (auto const& [filepath, buffer] : caches) {
            pool_mr.deallocate(buffer.data, buffer.size * sizeof(T));
        }
        caches.clear();
    }

    /**
     * @brief Sync the thread pool.
     * 
     */
    void sync() { thread_pool.reset(); }

    /**
     * @brief Save tensors to disk or cache.
     * 
     * @param tensors: Tensor map to be saved.
     * @param cache: Whether to cache the tensors in pinned memory.
     */
    void save(tensor_map_type& tensors, bool cache = false) {
        
        // Convert tensors to pinned memory
        std::vector<buffer_type> buffers;
        for (auto const& [filepath, tensor] : tensors) {
            // Allocate pinned host memory from pinned memory pool
            auto size = tensor.size();
            auto filesize = size * sizeof(T);
            auto buffer = buffer_type{static_cast<T*>(pool_mr.allocate(filesize)), size, filepath, tensor.shape()};

            // Copy tensor to pinned memory
            cudaMemcpy(buffer.data, tensor.raw_ptr(), filesize, cudaMemcpyDeviceToHost);
            buffers.push_back(buffer);
        }

        // Save buffers to cache or disk asynchronously
        for (const auto& buffer : buffers) {
            auto f = thread_pool.submit_task([this, buffer, cache]() {
                if (cache){
                    this -> caches[buffer.filepath] = buffer;
                } else {
                    std::vector<size_t> shape(buffer.shape.begin(), buffer.shape.end());
                    auto array = xt::adapt(buffer.data, buffer.size, xt::no_ownership(), shape); // This is adpated from the raw pointer, no ownership of the data.
                    xt::dump_npy(buffer.filepath, array);
                    this -> pool_mr.deallocate(buffer.data, buffer.size * sizeof(T)); // Need to deallocate the pinned memory after saving to disk.
                }
            });
        }
    }

    /**
     * @brief Load tensors from disk to host memory.
     * 
     * @param arrays: Array map to be loaded.
     */
    void load_from_disk_to_host(array_map_type& arrays) {
        for (const auto& pair : arrays) {
            const auto& filepath = pair.first;
            auto f = thread_pool.submit_task([this, &arrays, filepath]() {
                auto caches = this -> caches;
                if (caches.find(filepath) != caches.end()){
                    arrays[filepath] = caches[filepath];
                } else {
                    arrays[filepath] = std::make_shared<array_type>(xt::load_npy<T>(filepath)); // Load the array from disk to host memory. Use shared pointer for the xarray.
                }
            });
        }
    }

    /**
     * @brief Load tensors from host memory to device memory.
     * 
     * @param arrays: Array map to be loaded.
     * @return tensor_map_type: Tensor map loaded to device memory.
     */
    tensor_map_type load_from_host_to_device(array_map_type& arrays) {
        tensor_map_type tensors;
        for (const auto& pair : arrays) {
            const auto& filepath = pair.first;
            const auto& array = pair.second;
            if (std::holds_alternative<std::shared_ptr<array_type>>(array)) {
                auto ptr = std::get<std::shared_ptr<array_type>>(array);
                tensors[filepath] = vt::astensor<T, N>(*ptr);
            } else if (std::holds_alternative<buffer_type>(array)){
                auto buffer = std::get<buffer_type>(array);
                tensors[filepath] = vt::astensor<T, N>(buffer.data, buffer.shape);
            } else {
                throw std::runtime_error("No array or buffer found.");
            }
        }
        return tensors;
    }

    private:
        BS::thread_pool thread_pool;
        rmm::mr::pinned_memory_resource pinned_mr;
        rmm::mr::pool_memory_resource<rmm::mr::pinned_memory_resource> pool_mr;
        std::unordered_map<std::string, buffer_type> caches;
};


int main() {

    auto timer = CPUTimer();
    int num_files = 2;
    double filesize_gb = 1;
    size_t gb2b = 1024 * 1024 * 1024;
    size_t filesize_b = filesize_gb * gb2b;
    size_t num_elems = filesize_b / sizeof(float);
    size_t total_filesize = num_files * filesize_b;

    // Tensor
    std::unordered_map<std::string, vt::Tensor<float, 1>> tensors;
    auto tensor1 = vt::arange<float>(num_elems);
    auto tensor2 = vt::arange<float>(num_elems);
    tensors["/home/stsui/Documents/checkpoints/0.npy"] = tensor1;
    tensors["/home/stsui/Documents/checkpoints/1.npy"] = tensor2;

    // IOHandler
    auto io = IOHandler<float, 1>(total_filesize, total_filesize, num_files);

    timer.start();
    io.save(tensors);
    io.sync();
    auto duration = timer.stop();

    IOHandler<float, 1>::array_map_type arrays;
    arrays["/home/stsui/Documents/checkpoints/0.npy"] = std::monostate{};
    arrays["/home/stsui/Documents/checkpoints/1.npy"] = std::monostate{};

    timer.start();
    io.load_from_disk_to_host(arrays);
    io.sync();
    duration = timer.stop();
    std::cout << "Loading: " << duration << "ms" << std::endl;

    timer.start();
    auto tensors1 = io.load_from_host_to_device(arrays);
    duration = timer.stop();
    std::cout << "Host to Device " << duration << "ms" << std::endl;


    return 0;
}