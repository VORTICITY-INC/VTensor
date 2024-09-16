#include <gtest/gtest.h>

#include <lib/vtensor.hpp>

TEST(PowerBetweenTwoTensors, BasicAssertions) {
    auto tensor1 = vt::arange(6);
    auto tensor2 = vt::arange(6);
    auto re = vt::power(tensor1, tensor2);
    EXPECT_EQ(vt::asvector(re), (std::vector<float>{1, 1, 4, 27, 256, 3125}));
}

TEST(PowerBetweenTensorAndValue, BasicAssertions) {
    auto tensor1 = vt::arange(6);
    auto re = vt::power(tensor1, 2.0f);
    EXPECT_EQ(vt::asvector(re), (std::vector<float>{0, 1, 4, 9, 16, 25}));
}

TEST(PowerBetweenValueAndTensor, BasicAssertions) {
    auto tensor1 = vt::arange(6);
    auto re = vt::power(2.0f, tensor1);
    EXPECT_EQ(vt::asvector(re), (std::vector<float>{1, 2, 4, 8, 16, 32}));
}
