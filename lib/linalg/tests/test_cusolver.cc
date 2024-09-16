#include <cusolverDn.h>
#include <gtest/gtest.h>

#include <lib/vtensor.hpp>

TEST(GetGlobalCuSolverHandle, BasicAssertions) {
    auto handle1 = vt::cuda::cusolver.get_handle();
    auto handle2 = vt::cuda::cusolver.get_handle();
    EXPECT_EQ(handle1, handle2);
}

TEST(CreateNewCuSolverHandle, BasicAssertions) {
    auto handle1 = vt::cuda::cusolver.get_handle();
    auto handle2 = vt::cuda::create_cusolver_handle();
    EXPECT_NE(handle1, *handle2.get());
}
