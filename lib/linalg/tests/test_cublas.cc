#include <cublas_v2.h>
#include <gtest/gtest.h>

#include <lib/vtensor.hpp>

TEST(GetGlobalCuBLASHandle, BasicAssertions) {
    auto handle1 = vt::cuda::cublas.get_handle();
    auto handle2 = vt::cuda::cublas.get_handle();
    EXPECT_EQ(handle1, handle2);
}

TEST(CreateNewCuBLASHandle, BasicAssertions) {
    auto handle1 = vt::cuda::cublas.get_handle();
    auto handle2 = vt::cuda::create_cublas_handle();
    EXPECT_NE(handle1, *handle2.get());
}
