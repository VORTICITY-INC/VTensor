#include <iostream>
#include <lib/vtensor.hpp>
#include "xtensor/xarray.hpp"
#include <filesystem>
#include "BS_thread_pool.hpp" 
#include "xtensor/xio.hpp"
#include <xtensor/xview.hpp>
#include <chrono>
#include <xtensor/xrandom.hpp>
#include "xtensor/xview.hpp"
#include <cuda_runtime.h>
#include "cnpy.h"
#include <iostream>
#include <fstream>
#include <highfive/highfive.hpp>

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
    
    IOHandler(int num_threads) : pool(num_threads) {}
    
    ~IOHandler() { pool.wait(); }

    void sync() { pool.reset(); }

    array_map_type convert_to_xarray(tensor_map_type& tensors) {
        array_map_type arrays;
        for (auto const& [filepath, tensor] : tensors) {
            arrays[filepath] = std::make_shared<array_type>(vt::asxarray(tensor));
        }
        return arrays;
    }

    tensor_map_type convert_to_tensor(array_map_type& arrays) {
        tensor_map_type tensors;
        for (auto const& [filepath, array] : arrays) {
            tensors[filepath] = vt::astensor<T, N>(*array);
        }
        return tensors;
    }

    void save(tensor_map_type& tensors, bool cache = false) {
        auto arrays = convert_to_xarray(tensors);
        save(arrays, cache);
    }

    void save(array_map_type& arrays, bool cache = false) {
        for (const auto& pair : arrays) {
            const auto& filepath = pair.first;
            const auto& array = pair.second;
            if (cache) {
                caches[filepath] = array;
            } else{
                auto f = pool.submit_task([filepath, array]() {
                    xt::dump_npy(filepath, *array);
                });
            }
        }
    }

    void load(array_map_type& arrays) {
        for (const auto& pair : arrays) {
            const auto& filepath = pair.first;
            if (caches.find(filepath) != caches.end()){
                arrays[filepath] = caches[filepath];
            } else {
                auto f = pool.submit_task([&arrays, filepath]() {
                    arrays[filepath] = std::make_shared<array_type>(xt::load_npy<T>(filepath));
                });
            }
        }
    }

    private:
        BS::thread_pool pool;
        array_map_type caches;
};


int main() {    

    size_t size = 1000000000 * sizeof(float);

    auto timer = CPUTimer();
    auto io = IOHandler<float, 1>(2);
    std::unordered_map<std::string, vt::Tensor<float, 1>> tensors;

    auto tensor1 = vt::zeros<float>(1000000000);
    auto tensor2 = vt::zeros<float>(1000000000);
  
    // Allocate pinned host memory
    timer.start();
    float* h_tensor1;
    float* h_tensor2;
    cudaMallocHost(&h_tensor1, size);
    cudaMallocHost(&h_tensor2, size);

    auto duration = timer.stop();
    std::cout << "Allocating Duration: " << duration << " ms" << std::endl;

    timer.start();
    cudaMemcpy(h_tensor1, tensor1.raw_ptr(), size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_tensor2, tensor2.raw_ptr(), size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    duration = timer.stop();
    std::cout << "Transferring Duration: " << duration << " ms" << std::endl;
    
    // timer.start();
    // HighFive::File file("/home/stsui/Documents/checkpoints/out1.h5", HighFive::File::Overwrite);
    // HighFive::DataSet dataset1 = file.createDataSet<float>("/tensor1", HighFive::DataSpace({1000000000}));
    // dataset1.write(h_tensor1);
    // HighFive::DataSet dataset2 = file.createDataSet<float>("/tensor2", HighFive::DataSpace({1000000000}));
    // dataset2.write(h_tensor2);
    // duration = timer.stop();
    // std::cout << "Saving Duration: " << duration << " ms" << std::endl;

    timer.start();
    auto xarray1 = xt::adapt(h_tensor1, 1000000000, xt::no_ownership(), shape);


    
    // timer.start();
    // std::vector<size_t> shape = {1000000000};
    // auto xarray1 = xt::adapt(h_tensor1, 1000000000, xt::no_ownership(), shape);
    // auto xarray2 = xt::adapt(h_tensor2, 1000000000, xt::no_ownership(), shape);
    // xt::dump_npy("/home/stsui/Documents/checkpoints/0_0.npy", xarray1);
    // xt::dump_npy("/home/stsui/Documents/checkpoints/1_1.npy", xarray2);
    // duration = timer.stop();
    // std::cout << "Converting Duration: " << duration << " ms" << std::endl;
    


    cudaFreeHost(h_tensor1);
    cudaFreeHost(h_tensor2);
    // timer.start();
    // auto a1 = vt::asxarray<float>(tensor1);
    // auto a2 = vt::asxarray<float>(tensor2);
    // auto duration = timer.stop();
    // std::cout << "Converting Duration: " << duration << " ms" << std::endl;

    // tensors["/home/stsui/Documents/checkpoints/0_0.npy"] = tensor1;
    // tensors["/home/stsui/Documents/checkpoints/1_1.npy"] = tensor2;


    // timer.start();
    // io.save(tensors);
    // io.sync();
    // auto duration = timer.stop();
    // std::cout << "Saving Duration: " << duration << " ms" << std::endl;

    // timer.start();
    // std::unordered_map<std::string, std::shared_ptr<xt::xarray<float>>> arrays;
    // arrays["/home/stsui/Documents/checkpoints/0_0.npy"] = nullptr;
    // arrays["/home/stsui/Documents/checkpoints/1_1.npy"] = nullptr;
    // io.load(arrays);
    // io.sync();
    // duration = timer.stop();
    // std::cout << "Loading Duration: " << duration << " ms" << std::endl;
    // // std::cout << *arrays["/home/stsui/Documents/checkpoints/1_1.npy"] << std::endl;

    // timer.start();
    // auto tensors2 = io.convert_to_tensor(arrays);
    // duration = timer.stop();
    // std::cout << "Converting Duration: " << duration << " ms" << std::endl;

    // // vt::print(tensors2["/home/stsui/Documents/checkpoints/1_1.npy"]);


    // auto timer = CPUTimer();

    // timer.start();
    // auto arr1 = xt::load_npy<float>("/home/stsui/Documents/checkpoints/0_0.npy");
    // auto arr2 = xt::load_npy<float>("/home/stsui/Documents/checkpoints/1_1.npy");
    // auto duration = timer.stop();
    // std::cout << "(xtensor) Loading Duration: " << duration << " ms" << std::endl;

    // timer.start();
    // auto arr3 = cnpy::npy_load("/home/stsui/Documents/checkpoints/0_0.npy");
    // auto arr4 = cnpy::npy_load("/home/stsui/Documents/checkpoints/1_1.npy");
    // duration = timer.stop();
    // std::cout << "(cnpy) Loading Duration: " << duration << " ms" << std::endl;

    return 0;
}