#include <gtest/gtest.h>

#include <lib/vtensor.hpp>

TEST(allC, BasicAssertions) {
    auto tensor = vt::arange(12);
    EXPECT_EQ(vt::all(tensor), false);

    tensor = vt::ones(12);
    EXPECT_EQ(vt::all(tensor), true);

    tensor = vt::arange(12)({0, 12, 2});
    EXPECT_EQ(vt::all(tensor), false);

    tensor = vt::arange(12)({1, 12, 2});
    EXPECT_EQ(vt::all(tensor), true);

    auto tensor2 = vt::ones(12)({0, 12, 2}).reshape(2, 3);
    tensor2({0}, {0, 3}) = 0.0f;
    EXPECT_EQ(vt::asvector(vt::all(tensor2, 0)), (std::vector<bool>{0, 0, 0}));
    EXPECT_EQ(vt::asvector(vt::all(tensor2, 1)), (std::vector<bool>{0, 1}));
}

TEST(allF, BasicAssertions) {
    auto tensor = vt::arange(12, vt::Order::F);
    EXPECT_EQ(vt::all(tensor), false);

    tensor = vt::ones(12, vt::Order::F);
    EXPECT_EQ(vt::all(tensor), true);

    tensor = vt::arange(12, vt::Order::F)({0, 12, 2});
    EXPECT_EQ(vt::all(tensor), false);

    tensor = vt::arange(12, vt::Order::F)({1, 12, 2});
    EXPECT_EQ(vt::all(tensor), true);

    auto tensor2 = vt::ones(12, vt::Order::F)({0, 12, 2}).reshape(2, 3);
    tensor2({0}, {0, 3}) = 0.0f;
    EXPECT_EQ(vt::asvector(vt::all(tensor2, 0)), (std::vector<bool>{0, 0, 0}));
    EXPECT_EQ(vt::asvector(vt::all(tensor2, 1)), (std::vector<bool>{0, 1}));
}