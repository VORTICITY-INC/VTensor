#include <gtest/gtest.h>

#include <lib/vtensor.hpp>

TEST(Iterator1D, BasicAssertions) {
    auto tensor = vt::arange(12)({1, 12, 2});
    EXPECT_EQ(vt::asvector(tensor), (std::vector<float>{1, 3, 5, 7, 9, 11}));
}

TEST(Iterator2D, BasicAssertions) {
    auto tensor = vt::arange(12).reshape(4, 3)({1, 3, 1}, {0, 3, 2});
    EXPECT_EQ(vt::asvector(tensor), (std::vector<float>{3, 5, 6, 8}));
}

TEST(Iterator3D, BasicAssertions) {
    auto tensor = vt::arange(24).reshape(4, 3, 2)({1, 3, 1}, {0, 3, 2}, {1, 2, 1});
    EXPECT_EQ(vt::asvector(tensor), (std::vector<float>{7, 11, 13, 17}));
}
