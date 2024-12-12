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
#include <highfive/H5Easy.hpp>
// #include <kvikio/file_handle.hpp>

using array_type = xt::xarray<float>;
using array_map_type = std::unordered_map<std::string, std::shared_ptr<array_type>>;
int numfiles = 2;
double filesize_gb = 1;
size_t num_elems = filesize_gb * 1e9 / sizeof(float);
double total_size = numfiles * filesize_gb;

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

array_map_type create_date() {
    array_map_type arrays;
    for (size_t i = 0; i < numfiles; i++) {
        std::string filename = "/home/stsui/Documents/checkpoints/" + std::to_string(i);
        auto arr = std::make_shared<array_type>(xt::arange<float>({float(num_elems)}));
        arrays[filename] = arr;
    }
    return arrays;
}

void xarray_save(array_map_type& arrays) {
    BS::thread_pool pool(numfiles);
    auto timer = CPUTimer();
    for (const auto& pair : arrays) {
        const auto& filepath = pair.first;
        const auto& array = pair.second;
        auto f = pool.submit_task([filepath, array]() {
            auto fpath = filepath + ".npy";
            xt::dump_npy(fpath, *array);
        });
    }
    timer.start();
    pool.wait();
    auto duration = timer.stop();
    std::cout << "Saving (xtensor) Duration: " << total_size * 1e3 / duration << " GB/s" << std::endl;
}

void cnpy_save(array_map_type& arrays) {
    BS::thread_pool pool(numfiles);
    auto timer = CPUTimer();
    for (const auto& pair : arrays) {
        const auto& filepath = pair.first;
        const auto& array = pair.second;
        auto f = pool.submit_task([filepath, array]() {
            std::vector<size_t> shape = {num_elems};
            auto fpath = filepath + ".npy";
            cnpy::npy_save(fpath, (*array).data(), shape, "w");
        });
    }
    timer.start();
    pool.wait();
    auto duration = timer.stop();
    std::cout << "Saving (cnpy) Duration: " << total_size * 1e3 / duration << " GB/s" << std::endl;
}

void highfive_save(array_map_type& arrays){
    BS::thread_pool pool(numfiles);
    auto timer = CPUTimer();

    auto fpath = "/home/stsui/Documents/checkpoints/out.h5";
    H5Easy::File file(fpath, H5Easy::File::Overwrite);
    timer.start();
    for (const auto& pair : arrays) {
        const auto& filepath = pair.first;
        const auto& array = pair.second;
        H5Easy::dump(file, filepath, *array);
    //     auto f = pool.submit_task([filepath, array]() {
    //         auto fpath = filepath + ".h5";
    //         
            // H5Easy::dump(file, "/A", *array);
        // });
    }
    // pool.wait();
    auto duration = timer.stop();
    std::cout << "Saving (HighFive) Duration: " << total_size * 1e3 / duration << " GB/s" << std::endl;
}

int main() {    
    auto arrays = create_date();
    // xarray_save(arrays);
    // cnpy_save(arrays);
    highfive_save(arrays);

    // // size_t size = 1000000000 * sizeof(float);
    // auto timer = CPUTimer();
    // // auto io = IOHandler<float, 1>(2);
    // // std::unordered_map<std::string, vt::Tensor<float, 1>> tensors;

    // auto tensor1 = vt::arange<float>(num_elems);
    // auto tensor2 = vt::arange<float>(num_elems);
  
    // // Allocate pinned host memory
    // // timer.start();
    // float* h_tensor1;
    // float* h_tensor2;
    // auto size = num_elems * sizeof(float);
    // cudaMallocHost(&h_tensor1, size);
    // cudaMallocHost(&h_tensor2, size);

    // cudaMemcpy(h_tensor1, tensor1.raw_ptr(), size, cudaMemcpyDeviceToHost);
    // cudaMemcpy(h_tensor2, tensor2.raw_ptr(), size, cudaMemcpyDeviceToHost);
    // cudaDeviceSynchronize();

    // timer.start();
    // HighFive::File file("/home/stsui/Documents/checkpoints/out1.h5", HighFive::File::Overwrite);
    // HighFive::DataSet dataset1 = file.createDataSet<float>("/tensor1", HighFive::DataSpace({num_elems}));
    // dataset1.write(h_tensor1);
    // HighFive::DataSet dataset2 = file.createDataSet<float>("/tensor2", HighFive::DataSpace({num_elems}));
    // dataset2.write(h_tensor2);
    // auto duration = timer.stop();
    // std::cout << "Saving Duration: " << duration << " ms" << std::endl;


    // cudaFreeHost(h_tensor1);
    // cudaFreeHost(h_tensor2);

    // auto duration = timer.stop();
    // std::cout << "Allocating Duration: " << duration << " ms" << std::endl;

    // timer.start();

    // cudaDeviceSynchronize();
    // duration = timer.stop();
    // std::cout << "Transferring Duration: " << duration << " ms" << std::endl;
    
    // timer.start();
    // HighFive::File file("/home/stsui/Documents/checkpoints/out1.h5", HighFive::File::Overwrite);
    // HighFive::DataSet dataset1 = file.createDataSet<float>("/tensor1", HighFive::DataSpace({1000000000}));
    // dataset1.write(h_tensor1);
    // HighFive::DataSet dataset2 = file.createDataSet<float>("/tensor2", HighFive::DataSpace({1000000000}));
    // dataset2.write(h_tensor2);
    // duration = timer.stop();
    // std::cout << "Saving Duration: " << duration << " ms" << std::endl;

    // timer.start();
    // auto xarray1 = xt::adapt(h_tensor1, 1000000000, xt::no_ownership(), shape);


    
    // timer.start();
    // std::vector<size_t> shape = {1000000000};
    // auto xarray1 = xt::adapt(h_tensor1, 1000000000, xt::no_ownership(), shape);
    // auto xarray2 = xt::adapt(h_tensor2, 1000000000, xt::no_ownership(), shape);
    // xt::dump_npy("/home/stsui/Documents/checkpoints/0_0.npy", xarray1);
    // xt::dump_npy("/home/stsui/Documents/checkpoints/1_1.npy", xarray2);
    // duration = timer.stop();
    // std::cout << "Converting Duration: " << duration << " ms" << std::endl;
    



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