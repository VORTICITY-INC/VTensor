#include <benchmark/benchmark.h>
#include <thrust/device_vector.h>

#include <lib/vtensor.hpp>

const size_t N = 1e9;  // 4GB Tensor operations

static void ThrustSum(benchmark::State& state) {
    vt::time::GPUTimer timer;
    auto vec = thrust::device_vector<float>(N);
    for (auto _ : state) {
        timer.start();
        thrust::reduce(vec.begin(), vec.end(), 0, thrust::plus<float>());
        auto elapsed_time = timer.stop();
        state.SetIterationTime(elapsed_time * 1e-3);
    }
}

static void Tensor1DSum(benchmark::State& state) {
    vt::time::GPUTimer timer;
    vt::Tensor tensor = vt::ones(N)({0, N});
    for (auto _ : state) {
        timer.start();
        vt::sum(tensor);
        auto elapsed_time = timer.stop();
        state.SetIterationTime(elapsed_time * 1e-3);
    }
}

static void Tensor2DSum(benchmark::State& state) {
    vt::time::GPUTimer timer;
    vt::Tensor tensor = vt::ones(2, N / 2)({0, 2}, {0, N / 2});
    for (auto _ : state) {
        timer.start();
        vt::sum(tensor);
        auto elapsed_time = timer.stop();
        state.SetIterationTime(elapsed_time * 1e-3);
    }
}

static void Tensor3DSum(benchmark::State& state) {
    vt::time::GPUTimer timer;
    auto shape = vt::Shape<3>{2, 2, N / 4};
    vt::Tensor tensor = vt::ones(shape)({0, 2}, {0, 2}, {0, N / 4});
    for (auto _ : state) {
        timer.start();
        vt::sum(tensor);
        auto elapsed_time = timer.stop();
        state.SetIterationTime(elapsed_time * 1e-3);
    }
}

static void Tensor4DSum(benchmark::State& state) {
    vt::time::GPUTimer timer;
    auto shape = vt::Shape<4>{2, 2, 2, N / 8};
    vt::Tensor tensor = vt::ones(shape)(vt::Slice(0, 2), vt::Slice(0, 2), vt::Slice(0, 2), vt::Slice(0, N / 8));
    for (auto _ : state) {
        timer.start();
        vt::sum(tensor);
        auto elapsed_time = timer.stop();
        state.SetIterationTime(elapsed_time * 1e-3);
    }
}

__global__ void kernel1(vt::CuTensor<float, 1> tensor1, vt::CuTensor<float, 1> tensor2) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    tensor1[i] = tensor2[i];
}

static void TwoCuTensor1DIndex(benchmark::State& state) {
    vt::time::GPUTimer timer;
    auto tensor1 = vt::ones(N);
    auto tensor2 = vt::arange(N);
    for (auto _ : state) {
        timer.start();
        kernel1<<<1, N>>>(tensor1, tensor2);
        auto elapsed_time = timer.stop();
        state.SetIterationTime(elapsed_time * 1e-3);
    }
}

__global__ void kernel2(vt::CuTensor<float, 2> tensor1, vt::CuTensor<float, 2> tensor2) {
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    tensor1(i, j) = tensor2(i, j);
}

static void TwoCuTensor2DIndex(benchmark::State& state) {
    vt::time::GPUTimer timer;
    auto tensor1 = vt::ones(2, N / 2);
    auto tensor2 = vt::arange(N).reshape(2, N / 2);
    dim3 tpb(N / 2, 2);
    for (auto _ : state) {
        timer.start();
        kernel2<<<1, tpb>>>(tensor1, tensor2);
        auto elapsed_time = timer.stop();
        state.SetIterationTime(elapsed_time * 1e-3);
    }
}

__global__ void kernel3(vt::CuTensor<float, 3> tensor1, vt::CuTensor<float, 3> tensor2) {
    int k = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int i = threadIdx.z + blockIdx.z * blockDim.z;
    tensor1(i, j, k) = tensor2(i, j, k);
}

static void TwoCuTensor3DIndex(benchmark::State& state) {
    vt::time::GPUTimer timer;
    auto shape = vt::Shape<3>{2, 2, N / 4};
    auto tensor1 = vt::ones(shape);
    auto tensor2 = vt::arange(N).reshape(2, 2, N / 4);
    dim3 tpb(N / 4, 2, 2);
    for (auto _ : state) {
        timer.start();
        kernel3<<<1, tpb>>>(tensor1, tensor2);
        auto elapsed_time = timer.stop();
        state.SetIterationTime(elapsed_time * 1e-3);
    }
}

BENCHMARK(ThrustSum)->Unit(benchmark::kMillisecond)->UseManualTime();
BENCHMARK(Tensor1DSum)->Unit(benchmark::kMillisecond)->UseManualTime();
BENCHMARK(Tensor2DSum)->Unit(benchmark::kMillisecond)->UseManualTime();
BENCHMARK(Tensor3DSum)->Unit(benchmark::kMillisecond)->UseManualTime();
BENCHMARK(Tensor4DSum)->Unit(benchmark::kMillisecond)->UseManualTime();
BENCHMARK(TwoCuTensor1DIndex)->UseManualTime();
BENCHMARK(TwoCuTensor2DIndex)->UseManualTime();
BENCHMARK(TwoCuTensor3DIndex)->UseManualTime();

BENCHMARK_MAIN();
