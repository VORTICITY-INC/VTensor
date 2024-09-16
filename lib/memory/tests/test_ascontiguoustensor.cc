#include <gtest/gtest.h>

#include <lib/vtensor.hpp>

TEST(ConvertToContiguousTensor, BasicAssertions) {
    auto tensor = vt::arange(12)({0, 12, 2});
    auto tensor1 = vt::ascontiguoustensor(tensor);
    EXPECT_EQ(tensor.shape(), (std::array<size_t, 1>{6}));
    EXPECT_EQ(tensor.strides(), (std::array<size_t, 1>{2}));
    EXPECT_EQ(tensor1.shape(), (std::array<size_t, 1>{6}));
    EXPECT_EQ(tensor1.strides(), (std::array<size_t, 1>{1}));
}
