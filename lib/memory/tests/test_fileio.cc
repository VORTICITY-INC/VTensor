#include <gtest/gtest.h>

#include <lib/vtensor.hpp>

TEST(IO, BasicAssertions) {
    auto tensor = vt::arange(12);
    vt::save("test.npy", tensor);
    auto tensor1 = vt::load<float, 1>("test.npy");
    EXPECT_EQ(vt::asvector(tensor), vt::asvector(tensor1));
}
