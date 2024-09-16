#include <gtest/gtest.h>

#include <lib/vtensor.hpp>

TEST(EyeGenerator, BasicAssertions) {
    auto tensor = vt::eye(3, 2);
    EXPECT_EQ(vt::asvector(tensor), (std::vector<float>{1, 0, 0, 1, 0, 0}));

    tensor = vt::eye(2, 3);
    EXPECT_EQ(vt::asvector(tensor), (std::vector<float>{1, 0, 0, 0, 1, 0}));

    tensor = vt::eye(2);
    EXPECT_EQ(vt::asvector(tensor), (std::vector<float>{1, 0, 0, 1}));
}