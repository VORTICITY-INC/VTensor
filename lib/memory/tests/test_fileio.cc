#include <gtest/gtest.h>

#include <lib/vtensor.hpp>

TEST(IOC, BasicAssertions) {
    auto tensor = vt::arange(12).reshape(2, 2, 3);
    vt::save("test.npy", tensor);
    auto tensor1 = vt::load<float, 3, xt::layout_type::row_major>("test.npy");
    EXPECT_EQ(vt::asvector(tensor), vt::asvector(tensor1));
}

TEST(IOF, BasicAssertions) {
    auto tensor = vt::arange(12, vt::Order::F).reshape(2, 2, 3);
    vt::save("test.npy", tensor);
    auto tensor1 = vt::load<float, 3, xt::layout_type::column_major>("test.npy");
    EXPECT_EQ(vt::asvector(tensor), vt::asvector(tensor1));
}
