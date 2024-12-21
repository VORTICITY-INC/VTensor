#include <gtest/gtest.h>

#include <lib/vtensor.hpp>

TEST(whereC, BasicAssertions) {
    auto condition = vt::arange(24)({0, 24, 2}).reshape(2, 2, 3) > 12.0f;
    auto x = vt::arange(24)({0, 24, 2}).reshape(2, 2, 3);
    auto y = vt::ones(12).reshape(2, 2, 3);
    auto r = vt::where(condition, x, y);
    EXPECT_EQ(vt::asvector(r), (std::vector<float>{1, 1, 1, 1, 1, 1, 1, 14, 16, 18, 20, 22}));

    r = vt::where(condition, 0.0f, y);
    EXPECT_EQ(vt::asvector(r), (std::vector<float>{1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0}));

    r = vt::where(condition, x, 0.0f);
    EXPECT_EQ(vt::asvector(r), (std::vector<float>{0, 0, 0, 0, 0, 0, 0, 14, 16, 18, 20, 22}));
}

TEST(whereF, BasicAssertions) {
    auto condition = vt::arange(24, vt::Order::F)({0, 24, 2}).reshape(2, 2, 3) > 12.0f;
    auto x = vt::arange(24, vt::Order::F)({0, 24, 2}).reshape(2, 2, 3);
    auto y = vt::ones(12, vt::Order::F).reshape(2, 2, 3);
    auto r = vt::where(condition, x, y);
    EXPECT_EQ(vt::asvector(r), (std::vector<float>{1, 1, 1, 1, 1, 1, 1, 14, 16, 18, 20, 22}));

    r = vt::where(condition, 0.0f, y);
    EXPECT_EQ(vt::asvector(r), (std::vector<float>{1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0}));

    r = vt::where(condition, x, 0.0f);
    EXPECT_EQ(vt::asvector(r), (std::vector<float>{0, 0, 0, 0, 0, 0, 0, 14, 16, 18, 20, 22}));
}