#include <gtest/gtest.h>

#include <lib/vtensor.hpp>

TEST(CopyFromDeviceToDeviceC, BasicAssertions) {
    auto tensor = vt::arange(12)({1, 12, 2});
    auto tensor1 = vt::copy(tensor);
    EXPECT_EQ(vt::asvector(tensor1), (std::vector<float>{1, 3, 5, 7, 9, 11}));
}

TEST(CopyFromDeviceToDeviceF, BasicAssertions) {
    auto tensor = vt::arange(12, vt::Order::F)({1, 12, 2});
    auto tensor1 = vt::copy(tensor);
    EXPECT_EQ(vt::asvector(tensor1), (std::vector<float>{1, 3, 5, 7, 9, 11}));
}