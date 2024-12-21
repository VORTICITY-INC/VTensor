#include <gtest/gtest.h>

#include <lib/vtensor.hpp>

TEST(ZerosGeneratorC, BasicAssertions) {
    auto tensor1 = vt::zeros(vt::Shape<1>{12});
    EXPECT_EQ(vt::asvector(tensor1), (std::vector<float>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));

    auto tensor2 = vt::zeros(12);
    EXPECT_EQ(vt::asvector(tensor2), (std::vector<float>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));

    auto tensor3 = vt::zeros(3, 4);
    EXPECT_EQ(vt::asvector(tensor3), (std::vector<float>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
}

TEST(ZerosGeneratorF, BasicAssertions) {
    auto tensor1 = vt::zeros(vt::Shape<1>{12}, vt::Order::F);
    EXPECT_EQ(vt::asvector(tensor1), (std::vector<float>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));

    auto tensor2 = vt::zeros(12, vt::Order::F);
    EXPECT_EQ(vt::asvector(tensor2), (std::vector<float>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));

    auto tensor3 = vt::zeros(3, 4, vt::Order::F);
    EXPECT_EQ(vt::asvector(tensor3), (std::vector<float>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
}