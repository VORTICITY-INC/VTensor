#include <gtest/gtest.h>

#include <lib/vtensor.hpp>

TEST(Max, BasicAssertions) {
    auto tensor = vt::arange(12)({1, 12, 2});
    EXPECT_EQ(vt::asvector(vt::max(tensor)), std::vector<float>{11});
    EXPECT_EQ(vt::asvector(vt::max(tensor > 5.0f)), std::vector<bool>{true});
}

TEST(MaxAlongAxis, BasicAssertions) {
    auto tensor = vt::arange(72).reshape(6, 4, 3);
    auto tensor1 = tensor({2, 6, 2}, {0, 4, 2}, {0, 3});
    auto re = vt::max(tensor1, 0);
    EXPECT_EQ(vt::asvector(re), (std::vector<float>{48, 49, 50, 54, 55, 56}));

    re = vt::max(tensor1, 1);
    EXPECT_EQ(vt::asvector(re), (std::vector<float>{30, 31, 32, 54, 55, 56}));

    re = vt::max(tensor1, 2);
    EXPECT_EQ(vt::asvector(re), (std::vector<float>{26, 32, 50, 56}));

    auto re1 = vt::max(tensor1 > 40.0f, 0);
    EXPECT_EQ(vt::asvector(re1), (std::vector<bool>{1, 1, 1, 1, 1, 1}));

    re1 = vt::max(tensor1 > 40.0f, 1);
    EXPECT_EQ(vt::asvector(re1), (std::vector<bool>{0, 0, 0, 1, 1, 1}));

    re1 = vt::max(tensor1 > 40.0f, 2);
    EXPECT_EQ(vt::asvector(re1), (std::vector<bool>{0, 0, 1, 1}));
}
