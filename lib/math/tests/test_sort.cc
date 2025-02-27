#include <gtest/gtest.h>

#include <lib/vtensor.hpp>

TEST(SortC, BasicAssertions) {
    auto tensor = vt::arange(12)({1, 12, 2}).reshape(2, 3);
    auto re = vt::sort(tensor);
    EXPECT_EQ(vt::asvector(re), (std::vector<float>{1, 3, 5, 7, 9, 11}));

    tensor = vt::arange(12.0f, 0.0f, -1.0f)({1, 12, 2}).reshape(2, 3);
    re = vt::sort(tensor);
    EXPECT_EQ(vt::asvector(re), (std::vector<float>{1, 3, 5, 7, 9, 11}));
}

TEST(SortF, BasicAssertions) {
    auto tensor = vt::arange(12, vt::Order::F)({1, 12, 2}).reshape(2, 3);
    auto re = vt::sort(tensor);
    EXPECT_EQ(vt::asvector(re), (std::vector<float>{1, 3, 5, 7, 9, 11}));

    tensor = vt::arange(12.0f, 0.0f, -1.0f, vt::Order::F)({1, 12, 2}).reshape(2, 3);
    re = vt::sort(tensor);
    EXPECT_EQ(vt::asvector(re), (std::vector<float>{1, 3, 5, 7, 9, 11}));
}

TEST(SortlongAxisC, BasicAssertions) {
    auto tensor = vt::arange(12.0f, 0.0f, -1.0f).reshape(2, 2, 3);
    auto re = vt::sort(tensor, 0);
    EXPECT_EQ(vt::asvector(re), (std::vector<float>{6, 5, 4, 3, 2, 1, 12, 11, 10, 9, 8, 7}));

    re = vt::sort(tensor, 1);
    EXPECT_EQ(vt::asvector(re), (std::vector<float>{9, 8, 7, 12, 11, 10, 3, 2, 1, 6, 5, 4}));

    re = vt::sort(tensor, 2);
    EXPECT_EQ(vt::asvector(re), (std::vector<float>{10, 11, 12, 7, 8, 9, 4, 5, 6, 1, 2, 3}));
}

TEST(SortlongAxisF, BasicAssertions) {
    auto tensor = vt::arange(12.0f, 0.0f, -1.0f, vt::Order::F).reshape(2, 2, 3);
    auto re = vt::sort(tensor, 0);
    EXPECT_EQ(vt::asvector(re), (std::vector<float>{11, 12, 9, 10, 7, 8, 5, 6, 3, 4, 1, 2}));

    re = vt::sort(tensor, 1);
    EXPECT_EQ(vt::asvector(re), (std::vector<float>{10, 9, 12, 11, 6, 5, 8, 7, 2, 1, 4, 3}));

    re = vt::sort(tensor, 2);
    EXPECT_EQ(vt::asvector(re), (std::vector<float>{4, 3, 2, 1, 8, 7, 6, 5, 12, 11, 10, 9}));
}