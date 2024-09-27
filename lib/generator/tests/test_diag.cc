#include <gtest/gtest.h>

#include <lib/vtensor.hpp>

TEST(Test1DDiag, BasicAssertions) {
    auto x = vt::arange(6)({0, 6, 2});
    auto re = vt::diag(x);
    EXPECT_EQ(vt::asvector(re), (std::vector<float>{0, 0, 0, 0, 2, 0, 0, 0, 4}));

    re = vt::diag(x, 1);
    EXPECT_EQ(vt::asvector(re), (std::vector<float>{0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 4, 0, 0 ,0 ,0}));

    re = vt::diag(x, -1);
    EXPECT_EQ(vt::asvector(re), (std::vector<float>{0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0 ,4 ,0}));
}

TEST(Test2DDiag, BasicAssertions) {
    auto x = vt::arange(24)({0, 24, 2}).reshape(3, 4);
    auto re = vt::diag(x);
    EXPECT_EQ(vt::asvector(re), (std::vector<float>{0, 10 ,20}));

    re = vt::diag(x, 1);
    EXPECT_EQ(vt::asvector(re), (std::vector<float>{2, 12 ,22}));

    re = vt::diag(x, -1);
    EXPECT_EQ(vt::asvector(re), (std::vector<float>{8, 18}));
}