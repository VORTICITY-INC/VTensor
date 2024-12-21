#include <gtest/gtest.h>

#include <lib/vtensor.hpp>

TEST(SqrtForTensorC, BasicAssertions) {
    auto tensor = vt::arange(6);
    auto re = vt::power(tensor, 2.0f);
    re = vt::sqrt(re);
    EXPECT_EQ(vt::asvector(re), (std::vector<float>{0, 1, 2, 3, 4, 5}));
}

TEST(SqrtForTensorF, BasicAssertions) {
    auto tensor = vt::arange(6, vt::Order::F).reshape(2, 3);
    auto re = vt::power(tensor, 2.0f);
    re = vt::sqrt(re);
    EXPECT_EQ(vt::asvector(re), (std::vector<float>{0, 1, 2, 3, 4, 5}));
}
