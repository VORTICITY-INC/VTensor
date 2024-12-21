#include <gtest/gtest.h>

#include <lib/vtensor.hpp>

TEST(RandC, BasicAssertions) {
    // Set the seed and offset to the original state
    vt::cuda::set_seed(0, vt::cuda::curand.get_handle());
    vt::cuda::set_offset(0, vt::cuda::curand.get_handle());

    auto shape = vt::Shape<1>{10};
    auto tensor = vt::random::rand(shape);

    float* d_data;
    auto h_data = std::vector<float>(10);
    cudaMalloc(&d_data, 10 * sizeof(float));

    curandGenerator_t gen1;
    curandCreateGenerator(&gen1, CURAND_RNG_PSEUDO_DEFAULT);
    curandGenerateUniform(gen1, d_data, 10);
    cudaMemcpy(h_data.data(), d_data, 10 * sizeof(float), cudaMemcpyDeviceToHost);

    auto vec = vt::asvector(tensor);
    for (size_t i = 0; i < 10; i++) EXPECT_EQ(vec[i], h_data[i]);
    curandDestroyGenerator(gen1);
    cudaFree(d_data);
}

TEST(RandF, BasicAssertions) {
    // Set the seed and offset to the original state
    vt::cuda::set_seed(0, vt::cuda::curand.get_handle());
    vt::cuda::set_offset(0, vt::cuda::curand.get_handle());

    auto shape = vt::Shape<1>{10};
    auto tensor = vt::random::rand(shape, vt::Order::F);

    float* d_data;
    auto h_data = std::vector<float>(10);
    cudaMalloc(&d_data, 10 * sizeof(float));

    curandGenerator_t gen1;
    curandCreateGenerator(&gen1, CURAND_RNG_PSEUDO_DEFAULT);
    curandGenerateUniform(gen1, d_data, 10);
    cudaMemcpy(h_data.data(), d_data, 10 * sizeof(float), cudaMemcpyDeviceToHost);

    auto vec = vt::asvector(tensor);
    for (size_t i = 0; i < 10; i++) EXPECT_EQ(vec[i], h_data[i]);
    curandDestroyGenerator(gen1);
    cudaFree(d_data);
}

TEST(QusiRandC, BasicAssertions) {
    auto gen = vt::cuda::create_curand_handle<vt::cuda::SCRAMBLED_SOBOL32>(2);
    auto tensor = vt::random::rand(10, vt::Order::C, *gen.get());

    float* d_data;
    auto h_data = std::vector<float>(10);
    cudaMalloc(&d_data, 10 * sizeof(float));

    curandGenerator_t gen1;
    curandCreateGenerator(&gen1, CURAND_RNG_QUASI_SCRAMBLED_SOBOL32);
    curandSetQuasiRandomGeneratorDimensions(gen1, 2);
    curandGenerateUniform(gen1, d_data, 10);
    cudaMemcpy(h_data.data(), d_data, 10 * sizeof(float), cudaMemcpyDeviceToHost);
    auto vec = vt::asvector(tensor);
    for (size_t i = 0; i < 10; i++) EXPECT_EQ(vec[i], h_data[i]);

    curandDestroyGenerator(gen1);
    cudaFree(d_data);
}

TEST(QusiRandF, BasicAssertions) {
    auto gen = vt::cuda::create_curand_handle<vt::cuda::SCRAMBLED_SOBOL32>(2);
    auto tensor = vt::random::rand(10, vt::Order::F, *gen.get());

    float* d_data;
    auto h_data = std::vector<float>(10);
    cudaMalloc(&d_data, 10 * sizeof(float));

    curandGenerator_t gen1;
    curandCreateGenerator(&gen1, CURAND_RNG_QUASI_SCRAMBLED_SOBOL32);
    curandSetQuasiRandomGeneratorDimensions(gen1, 2);
    curandGenerateUniform(gen1, d_data, 10);
    cudaMemcpy(h_data.data(), d_data, 10 * sizeof(float), cudaMemcpyDeviceToHost);
    auto vec = vt::asvector(tensor);
    for (size_t i = 0; i < 10; i++) EXPECT_EQ(vec[i], h_data[i]);

    curandDestroyGenerator(gen1);
    cudaFree(d_data);
}

TEST(Rand1DC, BasicAssertions) {
    // Set the seed and offset to the original state
    vt::cuda::set_seed(0, vt::cuda::curand.get_handle());
    vt::cuda::set_offset(0, vt::cuda::curand.get_handle());

    // Generate from global
    auto shape = vt::Shape<1>{10};
    auto tensor1 = vt::random::rand(shape);

    // Generate from local
    auto gen = vt::cuda::create_curand_handle();
    auto tensor2 = vt::random::rand(10, vt::Order::C, *gen.get());
    EXPECT_EQ(vt::asvector(tensor1), vt::asvector(tensor2));
}

TEST(Rand1DF, BasicAssertions) {
    // Set the seed and offset to the original state
    vt::cuda::set_seed(0, vt::cuda::curand.get_handle());
    vt::cuda::set_offset(0, vt::cuda::curand.get_handle());

    // Generate from global
    auto shape = vt::Shape<1>{10};
    auto tensor1 = vt::random::rand(shape, vt::Order::F);

    // Generate from local
    auto gen = vt::cuda::create_curand_handle();
    auto tensor2 = vt::random::rand(10, vt::Order::F, *gen.get());
    EXPECT_EQ(vt::asvector(tensor1), vt::asvector(tensor2));
}

TEST(Rand2DC, BasicAssertions) {
    // Set the seed and offset to the original state
    vt::cuda::set_seed(0, vt::cuda::curand.get_handle());
    vt::cuda::set_offset(0, vt::cuda::curand.get_handle());

    // Generate from global
    auto shape = vt::Shape<1>{10};
    auto tensor1 = vt::random::rand(shape);

    // Generate from local
    auto gen = vt::cuda::create_curand_handle();
    auto tensor2 = vt::random::rand(2, 5, vt::Order::C, *gen.get());
    EXPECT_EQ(vt::asvector(tensor1), vt::asvector(tensor2));
}

TEST(Rand2DF, BasicAssertions) {
    // Set the seed and offset to the original state
    vt::cuda::set_seed(0, vt::cuda::curand.get_handle());
    vt::cuda::set_offset(0, vt::cuda::curand.get_handle());

    // Generate from global
    auto shape = vt::Shape<1>{10};
    auto tensor1 = vt::random::rand(shape, vt::Order::F);

    // Generate from local
    auto gen = vt::cuda::create_curand_handle();
    auto tensor2 = vt::random::rand(2, 5, vt::Order::F, *gen.get());
    EXPECT_EQ(vt::asvector(tensor1), vt::asvector(tensor2));
}