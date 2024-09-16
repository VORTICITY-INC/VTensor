#include <gtest/gtest.h>

#include <lib/vtensor.hpp>

TEST(Any, BasicAssertions) {
    auto tensor = vt::arange(12);
    EXPECT_EQ(vt::any(tensor), true);

    tensor = vt::zeros(12);
    EXPECT_EQ(vt::any(tensor), false);

    tensor = vt::arange(12)({0, 12, 2});
    EXPECT_EQ(vt::any(tensor), true);

    tensor = vt::arange(12)({0});
    EXPECT_EQ(vt::any(tensor), false);

    auto tensor2 = vt::ones(12)({0, 12, 2}).reshape(2, 3);
    tensor2({0}, {0, 3}) = 0.0f;
    EXPECT_EQ(vt::asvector(vt::any(tensor2, 0)), (std::vector<bool>{1, 1, 1}));
    EXPECT_EQ(vt::asvector(vt::any(tensor2, 1)), (std::vector<bool>{0, 1}));
}