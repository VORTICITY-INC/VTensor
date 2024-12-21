#include <gtest/gtest.h>

#include <lib/vtensor.hpp>

TEST(OnesGeneratorC, BasicAssertions) {
    auto tensor1 = vt::ones(vt::Shape<1>{12});
    EXPECT_EQ(vt::asvector(tensor1), (std::vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}));

    auto tensor2 = vt::ones(12);
    EXPECT_EQ(vt::asvector(tensor2), (std::vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}));

    auto tensor3 = vt::ones(3, 4);
    EXPECT_EQ(vt::asvector(tensor3), (std::vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}));
}

TEST(OnesGeneratorF, BasicAssertions) {
    auto tensor1 = vt::ones(vt::Shape<1>{12}, vt::Order::F);
    EXPECT_EQ(vt::asvector(tensor1), (std::vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}));

    auto tensor2 = vt::ones(12, vt::Order::F);
    EXPECT_EQ(vt::asvector(tensor2), (std::vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}));

    auto tensor3 = vt::ones(3, 4, vt::Order::F);
    EXPECT_EQ(vt::asvector(tensor3), (std::vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}));
}
