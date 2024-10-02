#include <gtest/gtest.h>

#include <lib/vtensor.hpp>

TEST(Mean, BasicAssertions) {
    auto tensor = vt::arange(12)({1, 12, 3});
    EXPECT_EQ(vt::asvector(vt::mean(tensor)), std::vector<double>{5.5});
    EXPECT_EQ(vt::asvector(vt::mean(tensor > 5.0f)), std::vector<double>{0.5});
}

TEST(MeanAlongAxis, BasicAssertions) {
    auto tensor = vt::arange(72).reshape(6, 4, 3);
    auto tensor1 = tensor({2, 6, 2}, {0, 4, 2}, {0, 3});
    auto re = vt::mean(tensor1, 0);
    EXPECT_EQ(vt::asvector(re), (std::vector<double>{36, 37, 38, 42, 43, 44}));

    re = vt::mean(tensor1, 1);
    EXPECT_EQ(vt::asvector(re), (std::vector<double>{27, 28, 29, 51, 52, 53}));

    re = vt::mean(tensor1, 2);
    EXPECT_EQ(vt::asvector(re), (std::vector<double>{25, 31, 49, 55}));

    re = vt::mean(tensor1 > 30.0f, 0);
    EXPECT_EQ(vt::asvector(re), (std::vector<double>{0.5, 0.5, 0.5, 0.5, 1.0, 1.0}));

    re = vt::mean(tensor1 > 30.0f, 1);
    EXPECT_EQ(vt::asvector(re), (std::vector<double>{0, 0.5, 0.5, 1.0, 1.0, 1.0}));

    re = vt::mean(tensor1 > 30.0f, 2);
    EXPECT_EQ(vt::asvector(re), (std::vector<double>{0, double(2) / 3, 1.0, 1.0}));
}
