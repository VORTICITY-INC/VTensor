#include <gtest/gtest.h>

#include <lib/vtensor.hpp>

TEST(SqrtForTensor, BasicAssertions) {
    auto tensor = vt::arange(6);
    auto re = vt::power(tensor, 2.0f);
    re = vt::sqrt(re);
    EXPECT_EQ(vt::asvector(re), (std::vector<float>{0, 1, 2, 3, 4, 5}));
}
