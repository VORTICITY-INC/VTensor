#include <gtest/gtest.h>

#include <lib/vtensor.hpp>

TEST(Iterator1DC, BasicAssertions) {
    auto tensor = vt::arange(12)({1, 12, 2});
    EXPECT_EQ(vt::asvector(tensor), (std::vector<float>{1, 3, 5, 7, 9, 11}));
}

TEST(Iterator2DC, BasicAssertions) {
    auto tensor = vt::arange(12).reshape(4, 3)({1, 3, 1}, {0, 3, 2});
    EXPECT_EQ(vt::asvector(tensor), (std::vector<float>{3, 5, 6, 8}));
}

TEST(Iterator3DC, BasicAssertions) {
    auto tensor = vt::arange(24).reshape(4, 3, 2)({1, 3, 1}, {0, 3, 2}, {1, 2, 1});
    EXPECT_EQ(vt::asvector(tensor), (std::vector<float>{7, 11, 13, 17}));
}

TEST(Iterator1DF, BasicAssertions) {
    auto tensor = vt::arange(12, vt::Order::F)({1, 12, 2});
    EXPECT_EQ(vt::asvector(tensor), (std::vector<float>{1, 3, 5, 7, 9, 11}));
}

TEST(Iterator2DF, BasicAssertions) {
    auto tensor = vt::arange(12, vt::Order::F).reshape(4, 3)({1, 3, 1}, {0, 3, 2});
    EXPECT_EQ(vt::asvector(tensor), (std::vector<float>{1, 2, 9, 10}));
}

TEST(Iterator3DF, BasicAssertions) {
    auto tensor = vt::arange(24, vt::Order::F).reshape(4, 3, 2)({1, 3, 1}, {0, 3, 2}, {1, 2, 1});
    EXPECT_EQ(vt::asvector(tensor), (std::vector<float>{13, 14, 21, 22}));
}