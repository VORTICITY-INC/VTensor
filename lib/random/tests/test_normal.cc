#include <gtest/gtest.h>

#include <lib/vtensor.hpp>

TEST(NormalC, BasicAssertions) {
    // Set the seed and offset to the original state
    vt::cuda::set_seed(0, vt::cuda::curand.get_handle());
    vt::cuda::set_offset(0, vt::cuda::curand.get_handle());

    auto shape = vt::Shape<1>{10};
    auto tensor = vt::random::normal(shape);

    float* d_data;
    auto h_data = std::vector<float>(10);
    cudaMalloc(&d_data, 10 * sizeof(float));

    curandGenerator_t gen1;
    curandCreateGenerator(&gen1, CURAND_RNG_PSEUDO_DEFAULT);
    curandGenerateNormal(gen1, d_data, 10, 0, 1);
    cudaMemcpy(h_data.data(), d_data, 10 * sizeof(float), cudaMemcpyDeviceToHost);

    auto vec = vt::asvector(tensor);
    for (size_t i = 0; i < 10; i++) EXPECT_EQ(vec[i], h_data[i]);
    curandDestroyGenerator(gen1);
    cudaFree(d_data);
}

TEST(NormalF, BasicAssertions) {
    // Set the seed and offset to the original state
    vt::cuda::set_seed(0, vt::cuda::curand.get_handle());
    vt::cuda::set_offset(0, vt::cuda::curand.get_handle());

    auto shape = vt::Shape<1>{10};
    auto tensor = vt::random::normal(shape, 0.0f, 1.0f, vt::Order::F);

    float* d_data;
    auto h_data = std::vector<float>(10);
    cudaMalloc(&d_data, 10 * sizeof(float));

    curandGenerator_t gen1;
    curandCreateGenerator(&gen1, CURAND_RNG_PSEUDO_DEFAULT);
    curandGenerateNormal(gen1, d_data, 10, 0, 1);
    cudaMemcpy(h_data.data(), d_data, 10 * sizeof(float), cudaMemcpyDeviceToHost);

    auto vec = vt::asvector(tensor);
    for (size_t i = 0; i < 10; i++) EXPECT_EQ(vec[i], h_data[i]);
    curandDestroyGenerator(gen1);
    cudaFree(d_data);
}

TEST(QusiNormalC, BasicAssertions) {
    auto gen = vt::cuda::create_curand_handle<vt::cuda::SCRAMBLED_SOBOL32>(2);
    auto tensor = vt::random::normal(10, 0.0f, 1.0f, vt::Order::C, *gen.get());

    float* d_data;
    auto h_data = std::vector<float>(10);
    cudaMalloc(&d_data, 10 * sizeof(float));

    curandGenerator_t gen1;
    curandCreateGenerator(&gen1, CURAND_RNG_QUASI_SCRAMBLED_SOBOL32);
    curandSetQuasiRandomGeneratorDimensions(gen1, 2);
    curandGenerateNormal(gen1, d_data, 10, 0.0, 1.0);
    cudaMemcpy(h_data.data(), d_data, 10 * sizeof(float), cudaMemcpyDeviceToHost);
    auto vec = vt::asvector(tensor);
    for (size_t i = 0; i < 10; i++) EXPECT_EQ(vec[i], h_data[i]);

    curandDestroyGenerator(gen1);
    cudaFree(d_data);
}

TEST(QusiNormalF, BasicAssertions) {
    auto gen = vt::cuda::create_curand_handle<vt::cuda::SCRAMBLED_SOBOL32>(2);
    auto tensor = vt::random::normal(10, 0.0f, 1.0f, vt::Order::F, *gen.get());

    float* d_data;
    auto h_data = std::vector<float>(10);
    cudaMalloc(&d_data, 10 * sizeof(float));

    curandGenerator_t gen1;
    curandCreateGenerator(&gen1, CURAND_RNG_QUASI_SCRAMBLED_SOBOL32);
    curandSetQuasiRandomGeneratorDimensions(gen1, 2);
    curandGenerateNormal(gen1, d_data, 10, 0.0, 1.0);
    cudaMemcpy(h_data.data(), d_data, 10 * sizeof(float), cudaMemcpyDeviceToHost);
    auto vec = vt::asvector(tensor);
    for (size_t i = 0; i < 10; i++) EXPECT_EQ(vec[i], h_data[i]);

    curandDestroyGenerator(gen1);
    cudaFree(d_data);
}

TEST(Normal1DC, BasicAssertions) {
    // Set the seed and offset to the original state
    vt::cuda::set_seed(0, vt::cuda::curand.get_handle());
    vt::cuda::set_offset(0, vt::cuda::curand.get_handle());

    // Generate from global
    auto shape = vt::Shape<1>{10};
    auto tensor1 = vt::random::normal(shape);

    // Generate from local
    auto gen = vt::cuda::create_curand_handle();
    auto tensor2 = vt::random::normal(10, 0.0f, 1.0f, vt::Order::C, *gen.get());
    EXPECT_EQ(vt::asvector(tensor1), vt::asvector(tensor2));
}

TEST(Normal1DF, BasicAssertions) {
    // Set the seed and offset to the original state
    vt::cuda::set_seed(0, vt::cuda::curand.get_handle());
    vt::cuda::set_offset(0, vt::cuda::curand.get_handle());

    // Generate from global
    auto shape = vt::Shape<1>{10};
    auto tensor1 = vt::random::normal(shape, 0.0f, 1.0f, vt::Order::F);

    // Generate from local
    auto gen = vt::cuda::create_curand_handle();
    auto tensor2 = vt::random::normal(10, 0.0f, 1.0f, vt::Order::F, *gen.get());
    EXPECT_EQ(vt::asvector(tensor1), vt::asvector(tensor2));
}

TEST(Normal2DC, BasicAssertions) {
    // Set the seed and offset to the original state
    vt::cuda::set_seed(0, vt::cuda::curand.get_handle());
    vt::cuda::set_offset(0, vt::cuda::curand.get_handle());

    // Generate from global
    auto shape = vt::Shape<1>{10};
    auto tensor1 = vt::random::normal(shape);

    // Generate from local
    auto gen = vt::cuda::create_curand_handle();
    auto tensor2 = vt::random::normal(2, 5, 0.0f, 1.0f, vt::Order::C, *gen.get());
    EXPECT_EQ(vt::asvector(tensor1), vt::asvector(tensor2));
}

TEST(Normal2DF, BasicAssertions) {
    // Set the seed and offset to the original state
    vt::cuda::set_seed(0, vt::cuda::curand.get_handle());
    vt::cuda::set_offset(0, vt::cuda::curand.get_handle());

    // Generate from global
    auto shape = vt::Shape<1>{10};
    auto tensor1 = vt::random::normal(shape, 0.0f, 1.0f, vt::Order::F);

    // Generate from local
    auto gen = vt::cuda::create_curand_handle();
    auto tensor2 = vt::random::normal(2, 5, 0.0f, 1.0f, vt::Order::F, *gen.get());
    EXPECT_EQ(vt::asvector(tensor1), vt::asvector(tensor2));
}
