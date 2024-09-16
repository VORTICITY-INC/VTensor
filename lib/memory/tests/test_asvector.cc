#include <gtest/gtest.h>

#include <lib/vtensor.hpp>

TEST(CopyFromDeviceToHost, BasicAssertions) {
    auto tensor = vt::arange(12);
    auto vector = vt::asvector(tensor);
    EXPECT_EQ(vector, (std::vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}));
}
