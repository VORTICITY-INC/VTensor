#include <gtest/gtest.h>

#include <lib/vtensor.hpp>

TEST(TestExpandLHS, BasicAssertions) {
    auto tensor = vt::arange(6);
    auto x = vt::expand_dims_lhs<float, 1, 2>(tensor);
    EXPECT_EQ(x.shape(), (vt::Shape<3>{1, 1, 6}));
    EXPECT_EQ(x.strides(), (vt::Shape<3>{0, 0, 1}));
    EXPECT_EQ(vt::asvector(x), (std::vector<float>{0, 1, 2, 3, 4, 5}));
}


TEST(TestExpandRHS, BasicAssertions) {
    auto tensor = vt::arange(6);
    auto x = vt::expand_dims_rhs<float, 1, 2>(tensor);
    EXPECT_EQ(x.shape(), (vt::Shape<3>{6, 1, 1}));
    EXPECT_EQ(x.strides(), (vt::Shape<3>{1, 0, 0}));
    EXPECT_EQ(vt::asvector(x), (std::vector<float>{0, 1, 2, 3, 4, 5}));
}