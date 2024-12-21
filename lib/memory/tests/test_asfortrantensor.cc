#include <gtest/gtest.h>

#include <lib/vtensor.hpp>

TEST(ConvertToFortranTensor, BasicAssertions) {
    auto tensor = vt::arange(12)({0, 12, 2}).reshape(2, 3);
    auto tensor1 = vt::asfortrantensor(tensor);
    EXPECT_EQ(tensor.shape(), (std::array<size_t, 2>{2, 3}));
    EXPECT_EQ(tensor.strides(), (std::array<size_t, 2>{3, 1}));
    EXPECT_EQ(tensor.order(), vt::Order::C);
    EXPECT_EQ(vt::asvector(tensor), (std::vector<float>{0, 2, 4, 6, 8, 10}));
    EXPECT_EQ(tensor1.shape(), (std::array<size_t, 2>{2, 3}));
    EXPECT_EQ(tensor1.strides(), (std::array<size_t, 2>{1, 2}));
    EXPECT_EQ(tensor1.order(), vt::Order::F);
    EXPECT_EQ(vt::asvector(tensor1), (std::vector<float>{0, 6, 2, 8, 4, 10}));
}