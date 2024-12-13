
#include "BS_thread_pool.hpp" 
#include <rmm/mr/host/pinned_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <iostream>
#include <lib/vtensor.hpp>

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


template <typename T, size_t N>
class IOHandler {
    public:

    using array_type = xt::xarray<T>;
    using tensor_type = vt::Tensor<T, N>;
    using array_map_type = std::unordered_map<std::string, std::shared_ptr<array_type>>;
    using tensor_map_type = std::unordered_map<std::string, tensor_type>;
    
    IOHandler(size_t pinned_pool_size, int num_threads) : thread_pool(num_threads), pool_mr(pinned_mr, 0, pinned_pool_size) {}

    private:
        BS::thread_pool thread_pool;
        rmm::mr::pinned_memory_resource pinned_mr;
        rmm::mr::pool_memory_resource<rmm::mr::pinned_memory_resource> pool_mr;
};


int main() {

    int num_files = 1;
    double filesize_gb = 1;
    size_t gb2b = 1024 * 1024 * 1024;
    size_t filesize_b = filesize_gb * gb2b;
    size_t num_elems = filesize_b / sizeof(float);

    std::cout << num_elems << std::endl;
    auto io = IOHandler(filesize_b, 1);





    // auto timer = CPUTimer();
    // // // Create a pinned memory resource
    // rmm::mr::pinned_memory_resource base_pinned_mr;
    // // Specify initial and maximum pool sizes (e.g., 128 MB and 1 GB)
    // constexpr std::size_t initial_pool_size = 0;
    // size_t maximum_pool_size = 1024 * 1024 * 1024;

    // // Pass the base memory resource and pool sizes to the pool memory resource constructor
    // rmm::mr::pool_memory_resource<rmm::mr::pinned_memory_resource> pool_mr(&base_pinned_mr, initial_pool_size, maximum_pool_size);

    // timer.start();
    // float* float_array = static_cast<float*>(pool_mr.allocate(num_floats * sizeof(float)));

    
    // auto duration = timer.stop();
    // std::cout << "Loading: " <<  duration << "ms" << std::endl;
    // pool_mr.deallocate(pinned_ptr, 1024 * 1024 * 1024);

    // timer.start();
    // pinned_ptr = pool_mr.allocate(1024 * 1024 * 1024);
    // duration = timer.stop();
    // pool_mr.deallocate(pinned_ptr, 1024 * 1024 * 1024);

    // std::cout << "Loading: " <<  duration << "ms" << std::endl;

    return 0;
}