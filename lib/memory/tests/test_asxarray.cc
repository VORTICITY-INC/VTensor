#include <gtest/gtest.h>

#include <lib/vtensor.hpp>

TEST(CopyFromDeviceToHostXArrayC, BasicAssertions) {
    auto tensor = vt::arange(12);
    auto arr = (xt::xarray<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    auto arr1 = vt::asxarray<float, 1, xt::layout_type::row_major>(tensor);
    EXPECT_EQ(arr, arr1);
}

TEST(CopyFromDeviceToHostXArrayF, BasicAssertions) {
    auto tensor = vt::arange(12, vt::Order::F);
    auto arr = (xt::xarray<float, xt::layout_type::column_major>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    auto arr1 = vt::asxarray<float, 1, xt::layout_type::column_major>(tensor);
    EXPECT_EQ(arr, arr1);
}