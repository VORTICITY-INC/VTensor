#include <gtest/gtest.h>

#include <lib/vtensor.hpp>

__global__ void kernel1(vt::CuTensor<float, 2> tensor) { tensor[threadIdx.x] += 1; }

__global__ void kernel2(vt::CuTensor<float, 2> tensor) {
    int col = threadIdx.x;
    int row = threadIdx.y;
    tensor(row, col) += 1;
}

TEST(CuTensorIndexOperator, BasicAssertions) {
    auto tensor = vt::arange(12).reshape(4, 3)({1, 3, 1}, {0, 3, 2});
    kernel1<<<1, 6>>>(tensor);
    EXPECT_EQ(vt::asvector(tensor), (std::vector<float>{4, 6, 7, 9}));
}

TEST(CuTensorBracketOperator, BasicAssertions) {
    auto tensor = vt::arange(12).reshape(4, 3)({1, 3, 1}, {0, 3, 2});
    dim3 tpb(2, 2);
    kernel2<<<1, tpb>>>(tensor);
    EXPECT_EQ(vt::asvector(tensor), (std::vector<float>{4, 6, 7, 9}));
}
