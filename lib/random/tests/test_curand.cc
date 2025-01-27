#include <gtest/gtest.h>

#include <lib/vtensor.hpp>

TEST(GetGlobalCuRandHandle, BasicAssertions) {
    auto handle1 = vt::cuda::CuRand::get_instance().get_handle();
    auto handle2 = vt::cuda::CuRand::get_instance().get_handle();
    EXPECT_EQ(handle1, handle2);
}

TEST(CreateNewCuRandHandle, BasicAssertions) {
    auto handle1 = vt::cuda::CuRand::get_instance().get_handle();
    auto handle2 = vt::cuda::create_curand_handle();
    EXPECT_NE(handle1, *handle2.get());
}

TEST(CreateCuRandHandle, BasicAssertions) {
    auto gen1 = vt::cuda::create_curand_handle();
    auto gen2 = vt::cuda::create_curand_handle<vt::cuda::XORWOW>();
    auto gen3 = vt::cuda::create_curand_handle<vt::cuda::MMRG32K3A>();
    auto gen4 = vt::cuda::create_curand_handle<vt::cuda::MTGP32>();
    auto gen5 = vt::cuda::create_curand_handle<vt::cuda::MT19937>();
    auto gen6 = vt::cuda::create_curand_handle<vt::cuda::PHILOX4_32_10>();
    auto gen7 = vt::cuda::create_curand_handle<vt::cuda::SOBOL32>(2);
    auto gen8 = vt::cuda::create_curand_handle<vt::cuda::SCRAMBLED_SOBOL32>(2);
    auto gen9 = vt::cuda::create_curand_handle<vt::cuda::SOBOL64>(2);
    auto gen10 = vt::cuda::create_curand_handle<vt::cuda::SCRAMBLED_SOBOL64>(2);
}

TEST(SetSeed, BasicAssertions) {
    auto gen = vt::cuda::create_curand_handle();
    vt::cuda::set_seed(10, *gen.get());
}

TEST(SetOffset, BasicAssertions) {
    auto gen = vt::cuda::create_curand_handle<vt::cuda::SOBOL32>();
    vt::cuda::set_offset(10, *gen.get());
}
