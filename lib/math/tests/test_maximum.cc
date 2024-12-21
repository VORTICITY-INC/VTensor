#include <gtest/gtest.h>

#include <lib/vtensor.hpp>

TEST(MaximumBetweenTwoTensorsC, BasicAssertions) {
    auto tensor1 = vt::zeros(6) + 5.0f;
    auto tensor2 = vt::arange(12)({1, 12, 2});
    auto re = vt::maximum(tensor1, tensor2);
    EXPECT_EQ(vt::asvector(re), (std::vector<float>{5, 5, 5, 7, 9, 11}));

    auto re1 = vt::maximum(tensor1[0], tensor2[0]);
    EXPECT_EQ(vt::asvector(re1), (std::vector<float>{5}));
}

TEST(MaximumBetweenTwoTensorsF, BasicAssertions) {
    auto tensor1 = (vt::zeros(6, vt::Order::F) + 5.0f).reshape(2, 3);
    auto tensor2 = vt::arange(12, vt::Order::F)({1, 12, 2}).reshape(2, 3);
    auto re = vt::maximum(tensor1, tensor2);
    EXPECT_EQ(vt::asvector(re), (std::vector<float>{5, 5, 5, 7, 9, 11}));

    auto re1 = vt::maximum(tensor1[0][0], tensor2[0][0]);
    EXPECT_EQ(vt::asvector(re1), (std::vector<float>{5}));
}

TEST(MaximumBetweenTensorAndValueC, BasicAssertions) {
    auto tensor1 = vt::arange(12)({1, 12, 2});
    auto re = vt::maximum(tensor1, 5.0f);
    EXPECT_EQ(vt::asvector(re), (std::vector<float>{5, 5, 5, 7, 9, 11}));

    re = vt::maximum(5.0f, tensor1);
    EXPECT_EQ(vt::asvector(re), (std::vector<float>{5, 5, 5, 7, 9, 11}));
}

TEST(MaximumBetweenTensorAndValueF, BasicAssertions) {
    auto tensor1 = vt::arange(12, vt::Order::F)({1, 12, 2}).reshape(2, 3);
    auto re = vt::maximum(tensor1, 5.0f);
    EXPECT_EQ(vt::asvector(re), (std::vector<float>{5, 5, 5, 7, 9, 11}));

    re = vt::maximum(5.0f, tensor1);
    EXPECT_EQ(vt::asvector(re), (std::vector<float>{5, 5, 5, 7, 9, 11}));
}
