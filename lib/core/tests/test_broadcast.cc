#include <gtest/gtest.h>

#include <lib/vtensor.hpp>

TEST(BroadcastTwoTensor, BasicAssertions) {
    auto tensor1 = vt::arange(12)({0, 12, 2}).reshape(1, 2, 3);
    auto tensor2 = vt::arange(2).reshape(2, 1, 1);
    auto [t1, t2] = vt::broadcast(tensor1, tensor2);
    EXPECT_EQ(vt::asvector(t1), (std::vector<float>{0, 2, 4, 6, 8, 10, 0, 2, 4, 6, 8, 10}));
    EXPECT_EQ(vt::asvector(t2), (std::vector<float>{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1}));
}

TEST(BroadcastToGivenShape, BasicAssertions) {
    auto tensor1 = vt::arange(12)({0, 12, 2}).reshape(1, 2, 3);
    auto tensor2 = vt::broadcast_to(tensor1, vt::Shape<3>{2, 2, 3});
    EXPECT_EQ(vt::asvector(tensor2), (std::vector<float>{0, 2, 4, 6, 8, 10, 0, 2, 4, 6, 8, 10}));
}
