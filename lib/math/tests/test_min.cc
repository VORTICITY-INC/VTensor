#include <gtest/gtest.h>

#include <lib/vtensor.hpp>

TEST(Min, BasicAssertionsC) {
    auto tensor = vt::arange(12)({1, 12, 2});
    EXPECT_EQ(vt::asvector(vt::min(tensor)), std::vector<float>{1});
    EXPECT_EQ(vt::asvector(vt::min(tensor > 5.0f)), std::vector<bool>{false});
}

TEST(Min, BasicAssertionsF) {
    auto tensor = vt::arange(12, vt::Order::F)({1, 12, 2});
    EXPECT_EQ(vt::asvector(vt::min(tensor)), std::vector<float>{1});
    EXPECT_EQ(vt::asvector(vt::min(tensor > 5.0f)), std::vector<bool>{false});
}

TEST(MinAlongAxisC, BasicAssertions) {
    auto tensor = vt::arange(72).reshape(6, 4, 3);
    auto tensor1 = tensor({2, 6, 2}, {0, 4, 2}, {0, 3});
    auto re = vt::min(tensor1, 0);
    EXPECT_EQ(vt::asvector(re), (std::vector<float>{24, 25, 26, 30, 31, 32}));

    re = vt::min(tensor1, 1);
    EXPECT_EQ(vt::asvector(re), (std::vector<float>{24, 25, 26, 48, 49, 50}));

    re = vt::min(tensor1, 2);
    EXPECT_EQ(vt::asvector(re), (std::vector<float>{24, 30, 48, 54}));

    auto re1 = vt::min(tensor1 > 40.0f, 0);
    EXPECT_EQ(vt::asvector(re1), (std::vector<bool>{0, 0, 0, 0, 0, 0}));

    re1 = vt::min(tensor1 > 40.0f, 1);
    EXPECT_EQ(vt::asvector(re1), (std::vector<bool>{0, 0, 0, 1, 1, 1}));

    re1 = vt::min(tensor1 > 40.0f, 2);
    EXPECT_EQ(vt::asvector(re1), (std::vector<bool>{0, 0, 1, 1}));
}

TEST(MinAlongAxisF, BasicAssertions) {
    auto tensor = vt::arange(72, vt::Order::F).reshape(6, 4, 3);
    auto tensor1 = tensor({2, 6, 2}, {0, 4, 2}, {0, 3});
    auto re = vt::min(tensor1, 0);
    EXPECT_EQ(vt::asvector(re), (std::vector<float>{2, 14, 26, 38, 50, 62}));

    re = vt::min(tensor1, 1);
    EXPECT_EQ(vt::asvector(re), (std::vector<float>{2, 4, 26, 28, 50, 52}));

    re = vt::min(tensor1, 2);
    EXPECT_EQ(vt::asvector(re), (std::vector<float>{2, 4, 14, 16}));

    auto re1 = vt::min(tensor1 > 40.0f, 0);
    EXPECT_EQ(vt::asvector(re1), (std::vector<bool>{0, 0, 0, 0, 1, 1}));

    re1 = vt::min(tensor1 > 40.0f, 1);
    EXPECT_EQ(vt::asvector(re1), (std::vector<bool>{0, 0, 0, 0, 1, 1}));

    re1 = vt::min(tensor1 > 40.0f, 2);
    EXPECT_EQ(vt::asvector(re1), (std::vector<bool>{0, 0, 0, 0}));
}