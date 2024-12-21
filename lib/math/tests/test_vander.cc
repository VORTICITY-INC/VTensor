#include <gtest/gtest.h>

#include <lib/vtensor.hpp>

TEST(VanderC, BasicAssertions) {
    auto tensor = vt::arange(6);
    auto v = vt::vander(tensor, 2);
    EXPECT_EQ(vt::asvector(v), (std::vector<float>{0, 0, 1, 1, 1, 1, 4, 2, 1, 9, 3, 1, 16, 4, 1, 25, 5, 1}));
}

TEST(VanderF, BasicAssertions) {
    auto tensor = vt::arange(6, vt::Order::F);
    auto v = vt::vander(tensor, 2);
    EXPECT_EQ(vt::asvector(v), (std::vector<float>{0, 1, 4, 9, 16, 25, 0, 1, 2, 3, 4, 5, 1, 1, 1, 1, 1, 1}));
}