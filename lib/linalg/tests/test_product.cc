#include <gtest/gtest.h>

#include <lib/vtensor.hpp>

TEST(Dot, BasicAssertions) {
    auto t1 = vt::arange(12)({1, 12, 2});
    auto t2 = vt::arange(12)({0, 12, 2});
    auto re = vt::dot(t1, t2);
    EXPECT_EQ(re, 250);

    re = vt::dot(t1[1], t2[1]);
    EXPECT_EQ(re, 6);
}
