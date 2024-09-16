#include <gtest/gtest.h>

#include <lib/vtensor.hpp>

TEST(MaximumBetweenTwoTensors, BasicAssertions) {
    auto tensor1 = vt::zeros(6) + 5.0f;
    auto tensor2 = vt::arange(12)({1, 12, 2});
    auto re = vt::maximum(tensor1, tensor2);
    EXPECT_EQ(vt::asvector(re), (std::vector<float>{5, 5, 5, 7, 9, 11}));
}

TEST(MaximumBetweenTensorAndValue, BasicAssertions) {
    auto tensor1 = vt::arange(12)({1, 12, 2});
    auto re = vt::maximum(tensor1, 5.0f);
    EXPECT_EQ(vt::asvector(re), (std::vector<float>{5, 5, 5, 7, 9, 11}));

    re = vt::maximum(5.0f, tensor1);
    EXPECT_EQ(vt::asvector(re), (std::vector<float>{5, 5, 5, 7, 9, 11}));
}
