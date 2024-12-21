#include <gtest/gtest.h>

#include <lib/vtensor.hpp>

TEST(MinimumBetweenTwoTensorsC, BasicAssertions) {
    auto tensor1 = vt::zeros(6) + 5.0f;
    auto tensor2 = vt::arange(12)({1, 12, 2});
    auto re = vt::minimum(tensor1, tensor2);
    EXPECT_EQ(vt::asvector(re), (std::vector<float>{1, 3, 5, 5, 5, 5}));

    auto re1 = vt::minimum(tensor1[0], tensor2[0]);
    EXPECT_EQ(vt::asvector(re1), (std::vector<float>{1}));
}

TEST(MinimumBetweenTwoTensorsF, BasicAssertions) {
    auto tensor1 = (vt::zeros(6, vt::Order::F) + 5.0f).reshape(2, 3);
    auto tensor2 = vt::arange(12, vt::Order::F)({1, 12, 2}).reshape(2, 3);
    auto re = vt::minimum(tensor1, tensor2);
    EXPECT_EQ(vt::asvector(re), (std::vector<float>{1, 3, 5, 5, 5, 5}));

    auto re1 = vt::minimum(tensor1[0][0], tensor2[0][0]);
    EXPECT_EQ(vt::asvector(re1), (std::vector<float>{1}));
}

TEST(MinimumBetweenTensorAndValueC, BasicAssertions) {
    auto tensor1 = vt::arange(12)({1, 12, 2});
    auto re = vt::minimum(tensor1, 5.0f);
    EXPECT_EQ(vt::asvector(re), (std::vector<float>{1, 3, 5, 5, 5, 5}));

    re = vt::minimum(5.0f, tensor1);
    EXPECT_EQ(vt::asvector(re), (std::vector<float>{1, 3, 5, 5, 5, 5}));
}

TEST(MinimumBetweenTensorAndValueF, BasicAssertions) {
    auto tensor1 = vt::arange(12)({1, 12, 2}).reshape(2, 3);
    auto re = vt::minimum(tensor1, 5.0f);
    EXPECT_EQ(vt::asvector(re), (std::vector<float>{1, 3, 5, 5, 5, 5}));

    re = vt::minimum(5.0f, tensor1);
    EXPECT_EQ(vt::asvector(re), (std::vector<float>{1, 3, 5, 5, 5, 5}));
}
