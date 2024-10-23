#include <gtest/gtest.h>

#include <lib/vtensor.hpp>

TEST(Vander, BasicAssertions) {
    auto tensor = vt::arange(6);
    auto v = vt::vander(tensor, 2);
    EXPECT_EQ(vt::asvector(v), (std::vector<float>{0, 0, 1, 1, 1, 1, 4, 2, 1, 9, 3, 1, 16, 4, 1, 25, 5, 1}));
}