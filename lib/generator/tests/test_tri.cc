#include <gtest/gtest.h>

#include <lib/vtensor.hpp>

TEST(TestTri, BasicAssertions) {
    auto tensor = vt::tri(2, 3);
    EXPECT_EQ(vt::asvector(tensor), (std::vector<float>{1, 0, 0, 1, 1, 0}));

    tensor = vt::tri(2, 3, -1);
    EXPECT_EQ(vt::asvector(tensor), (std::vector<float>{0, 0, 0, 1, 0, 0}));

    tensor = vt::tri(2, 3, 1);
    EXPECT_EQ(vt::asvector(tensor), (std::vector<float>{1, 1, 0, 1, 1, 1}));

}

TEST(TestTril, BasicAssertions) {
    auto tensor = vt::arange(24)({0, 24, 2}).reshape(2, 2, 3);
    auto re = vt::tril(tensor);
    EXPECT_EQ(vt::asvector(re), (std::vector<float>{0, 0, 0, 6, 8, 0, 12, 0, 0, 18, 20, 0}));

    re = vt::tril(tensor, 1);
    EXPECT_EQ(vt::asvector(re), (std::vector<float>{0, 2, 0, 6, 8, 10, 12, 14, 0, 18, 20, 22}));

    re = vt::tril(tensor, -1);
    EXPECT_EQ(vt::asvector(re), (std::vector<float>{0, 0, 0, 6, 0, 0, 0, 0, 0, 18, 0, 0}));

    auto tensor1 = vt::arange(24)({0, 24, 2}).reshape(3, 4);
    auto re1 = vt::tril(tensor1);
    EXPECT_EQ(vt::asvector(re1), (std::vector<float>{0, 0, 0, 0, 8, 10, 0, 0, 16, 18, 20, 0}));

    re1 = vt::tril(tensor1, 1);
    EXPECT_EQ(vt::asvector(re1), (std::vector<float>{0, 2, 0, 0, 8, 10, 12, 0, 16, 18, 20, 22}));

    re1 = vt::tril(tensor1, -1);
    EXPECT_EQ(vt::asvector(re1), (std::vector<float>{0, 0, 0, 0, 8, 0, 0, 0, 16, 18, 0, 0}));
}

TEST(TestTriu, BasicAssertions) {
    auto tensor = vt::arange(24)({0, 24, 2}).reshape(2, 2, 3);
    auto re = vt::triu(tensor);
    EXPECT_EQ(vt::asvector(re), (std::vector<float>{0, 2, 4, 0, 8, 10, 12, 14, 16, 0, 20, 22}));

    re = vt::triu(tensor, 1);
    EXPECT_EQ(vt::asvector(re), (std::vector<float>{0, 2, 4, 0, 0, 10, 0, 14, 16, 0, 0, 22}));

    re = vt::triu(tensor, -1);
    EXPECT_EQ(vt::asvector(re), (std::vector<float>{0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22}));

    auto tensor1 = vt::arange(24)({0, 24, 2}).reshape(3, 4);
    auto re1 = vt::triu(tensor1);
    EXPECT_EQ(vt::asvector(re1), (std::vector<float>{0, 2, 4, 6, 0, 10, 12, 14, 0, 0, 20, 22}));

    re1 = vt::triu(tensor1, 1);
    EXPECT_EQ(vt::asvector(re1), (std::vector<float>{0, 2, 4, 6, 0, 0, 12, 14, 0, 0, 0, 22}));

    re1 = vt::triu(tensor1, -1);
    EXPECT_EQ(vt::asvector(re1), (std::vector<float>{0, 2, 4, 6, 8, 10, 12, 14, 0, 18, 20, 22}));
}
