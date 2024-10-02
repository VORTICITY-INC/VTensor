#include <gtest/gtest.h>

#include <lib/vtensor.hpp>

TEST(MinimumBetweenTwoTensors, BasicAssertions) {
    auto tensor1 = vt::zeros(6) + 5.0f;
    auto tensor2 = vt::arange(12)({1, 12, 2});
    auto re = vt::minimum(tensor1, tensor2);
    EXPECT_EQ(vt::asvector(re), (std::vector<float>{1, 3, 5, 5, 5, 5}));

    auto re1 = vt::minimum(tensor1[0], tensor2[0]);
    EXPECT_EQ(vt::asvector(re1), (std::vector<float>{1}));
}

TEST(MinimumBetweenTensorAndValue, BasicAssertions) {
    auto tensor1 = vt::arange(12)({1, 12, 2});
    auto re = vt::minimum(tensor1, 5.0f);
    EXPECT_EQ(vt::asvector(re), (std::vector<float>{1, 3, 5, 5, 5, 5}));

    re = vt::minimum(5.0f, tensor1);
    EXPECT_EQ(vt::asvector(re), (std::vector<float>{1, 3, 5, 5, 5, 5}));
}
