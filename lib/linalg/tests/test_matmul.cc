#include <gtest/gtest.h>

#include <lib/vtensor.hpp>

TEST(Matmul1D, BasicAssertions) {
    auto tensor1 = vt::arange(12)({1, 12, 2});
    auto tensor2 = vt::arange(12)({0, 12, 2});
    auto re = vt::matmul(tensor1, tensor2);
    EXPECT_EQ(vt::asvector(re), (std::vector<float>{250}));
}

TEST(Matmul2D, BasicAssertions) {
    auto tensor1 = vt::arange(6).reshape(2, 3)({0}, {0, 3});
    auto tensor2 = vt::arange(12).reshape(3, 4);
    auto re = vt::matmul(tensor1, tensor2);
    EXPECT_EQ(vt::asvector(re), (std::vector<float>{20, 23, 26, 29}));
}

TEST(Matmul3D, BasicAssertions) {
    auto tensor1 = vt::arange(12).reshape(2, 2, 3)({0}, {0, 2}, {0, 3});
    auto tensor2 = vt::arange(24).reshape(2, 3, 4)({0}, {0, 3}, {0, 4});
    auto re = vt::matmul(tensor1, tensor2);
    EXPECT_EQ(vt::asvector(re), (std::vector<float>{20, 23, 26, 29, 56, 68, 80, 92}));
}