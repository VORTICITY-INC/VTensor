#include <gtest/gtest.h>

#include <lib/vtensor.hpp>

TEST(CopyFromHostToDevice, BasicAssertions) {
    std::vector<float> vector{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    auto tensor = vt::astensor(vector);
    EXPECT_EQ(vt::asvector(tensor), vector);
    auto tensor1 = vt::astensor(vector.data(), vector.size());
    EXPECT_EQ(vt::asvector(tensor1), vector);
    auto arr = xt::xarray<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    auto tensor2 = vt::astensor<float, 1>(arr);
    EXPECT_EQ(vt::asxarray(tensor2), arr);
}