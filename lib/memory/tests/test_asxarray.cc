#include <gtest/gtest.h>

#include <lib/vtensor.hpp>

TEST(CopyFromDeviceToHostXArray, BasicAssertions) {
    auto tensor = vt::arange(12);
    auto arr = vt::asxarray(tensor);
    EXPECT_EQ(arr, (xt::xarray<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}));
}
