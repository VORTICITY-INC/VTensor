#include <gtest/gtest.h>

#include <lib/vtensor.hpp>

TEST(Sum, BasicAssertions) {
    auto tensor = vt::arange(12)({1, 12, 2});
    EXPECT_EQ(vt::asvector(vt::sum(tensor)), std::vector<float>{36});
    EXPECT_EQ(vt::asvector(vt::sum(tensor > 5.0f)), std::vector<int>{3});
}

TEST(SumAlongAxis, BasicAssertions) {
    auto tensor = vt::arange(72).reshape(6, 4, 3);
    auto tensor1 = tensor({2, 6, 2}, {0, 4, 2}, {0, 3});
    auto re = vt::sum(tensor1, 0);
    EXPECT_EQ(vt::asvector(re), (std::vector<float>{72, 74, 76, 84, 86, 88}));

    re = vt::sum(tensor1, 1);
    EXPECT_EQ(vt::asvector(re), (std::vector<float>{54, 56, 58, 102, 104, 106}));

    re = vt::sum(tensor1, 2);
    EXPECT_EQ(vt::asvector(re), (std::vector<float>{75, 93, 147, 165}));

    auto re1 = vt::sum(tensor1 > 30.0f, 0);
    EXPECT_EQ(vt::asvector(re1), (std::vector<int>{1, 1, 1, 1, 2, 2}));

    re1 = vt::sum(tensor1 > 30.0f, 1);
    EXPECT_EQ(vt::asvector(re1), (std::vector<int>{0, 1, 1, 2, 2, 2}));

    re1 = vt::sum(tensor1 > 30.0f, 2);
    EXPECT_EQ(vt::asvector(re1), (std::vector<int>{0, 2, 3, 3}));
}
