#include <gtest/gtest.h>

#include <lib/vtensor.hpp>

TEST(CopyFromHostToDeviceC, BasicAssertions) {
    std::vector<float> vector{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    auto tensor = vt::astensor(vector);
    EXPECT_EQ(vt::asvector(tensor), vector);
    auto tensor1 = vt::astensor(vector.data(), vector.size());
    EXPECT_EQ(vt::asvector(tensor1), vector);
    auto arr = xt::xarray<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    auto tensor2 = vt::astensor<float, 1>(arr);
    auto arr1 = vt::asxarray<float, 1, xt::layout_type::row_major>(tensor2);
    EXPECT_EQ(arr, arr1);
}

TEST(CopyFromHostToDeviceF, BasicAssertions) {
    std::vector<float> vector{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    auto tensor = vt::astensor(vector, vt::Order::F);
    EXPECT_EQ(vt::asvector(tensor), vector);
    auto tensor1 = vt::astensor(vector.data(), vector.size(), vt::Order::F);
    EXPECT_EQ(vt::asvector(tensor1), vector);
    auto arr = xt::xarray<float, xt::layout_type::column_major>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    auto tensor2 = vt::astensor<float, 1>(arr);
    auto arr2 = vt::asxarray<float, 1, xt::layout_type::column_major>(tensor2);
    EXPECT_EQ(arr, arr2);
}