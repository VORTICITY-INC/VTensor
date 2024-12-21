#include <gtest/gtest.h>

#include <lib/vtensor.hpp>

TEST(TransposeC, BasicAssertions) {
    auto tensor = vt::arange(12).reshape(3, 2, 2);
    auto tensor1 = vt::transpose(tensor);
    EXPECT_EQ(tensor.size(), 12);
    EXPECT_EQ(tensor.shape(), (std::array<size_t, 3>{3, 2, 2}));
    EXPECT_EQ(tensor.strides(), (std::array<size_t, 3>{4, 2, 1}));
    EXPECT_EQ(tensor.start(), 0);
    EXPECT_EQ(tensor.order(), vt::Order::C);
    EXPECT_EQ(tensor.contiguous(), true);
    EXPECT_EQ(vt::asvector(tensor), (std::vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}));

    EXPECT_EQ(tensor1.size(), 12);
    EXPECT_EQ(tensor1.shape(), (std::array<size_t, 3>{2, 2, 3}));
    EXPECT_EQ(tensor1.strides(), (std::array<size_t, 3>{1, 2, 4}));
    EXPECT_EQ(tensor1.start(), 0);
    EXPECT_EQ(tensor1.order(), vt::Order::C);
    EXPECT_EQ(tensor1.contiguous(), false);
    EXPECT_EQ(vt::asvector(tensor), (std::vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}));
    EXPECT_EQ(vt::asvector(tensor1), (std::vector<float>{0, 4, 8, 2, 6, 10, 1, 5, 9, 3, 7, 11}));
}

TEST(TransposeF, BasicAssertions) {
    auto tensor = vt::arange(12, vt::Order::F).reshape(3, 2, 2);
    auto tensor1 = vt::transpose(tensor);
    EXPECT_EQ(tensor.size(), 12);
    EXPECT_EQ(tensor.shape(), (std::array<size_t, 3>{3, 2, 2}));
    EXPECT_EQ(tensor.strides(), (std::array<size_t, 3>{1, 3, 6}));
    EXPECT_EQ(tensor.start(), 0);
    EXPECT_EQ(tensor.order(), vt::Order::F);
    EXPECT_EQ(tensor.contiguous(), true);
    EXPECT_EQ(vt::asvector(tensor), (std::vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}));

    EXPECT_EQ(tensor1.size(), 12);
    EXPECT_EQ(tensor1.shape(), (std::array<size_t, 3>{2, 2, 3}));
    EXPECT_EQ(tensor1.strides(), (std::array<size_t, 3>{6, 3, 1}));
    EXPECT_EQ(tensor1.start(), 0);
    EXPECT_EQ(tensor1.order(), vt::Order::F);
    EXPECT_EQ(tensor1.contiguous(), false);
    EXPECT_EQ(vt::asvector(tensor), (std::vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}));
    EXPECT_EQ(vt::asvector(tensor1), (std::vector<float>{0, 6, 3, 9, 1, 7, 4, 10, 2, 8, 5, 11}));
}

TEST(TransposeWithAxisC, BasicAssertions) {
    auto tensor = vt::arange(12).reshape(3, 2, 2);
    auto tensor1 = vt::transpose(tensor, {2, 1, 0});

    EXPECT_EQ(tensor.size(), 12);
    EXPECT_EQ(tensor.shape(), (std::array<size_t, 3>{3, 2, 2}));
    EXPECT_EQ(tensor.strides(), (std::array<size_t, 3>{4, 2, 1}));
    EXPECT_EQ(tensor.start(), 0);
    EXPECT_EQ(tensor.order(), vt::Order::C);
    EXPECT_EQ(tensor.contiguous(), true);
    EXPECT_EQ(vt::asvector(tensor), (std::vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}));

    EXPECT_EQ(tensor1.size(), 12);
    EXPECT_EQ(tensor1.shape(), (std::array<size_t, 3>{2, 2, 3}));
    EXPECT_EQ(tensor1.strides(), (std::array<size_t, 3>{1, 2, 4}));
    EXPECT_EQ(tensor1.start(), 0);
    EXPECT_EQ(tensor1.order(), vt::Order::C);
    EXPECT_EQ(tensor1.contiguous(), false);
    EXPECT_EQ(vt::asvector(tensor), (std::vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}));
    EXPECT_EQ(vt::asvector(tensor1), (std::vector<float>{0, 4, 8, 2, 6, 10, 1, 5, 9, 3, 7, 11}));
}

TEST(TransposeWithAxisF, BasicAssertions) {
    auto tensor = vt::arange(12, vt::Order::F).reshape(3, 2, 2);
    auto tensor1 = vt::transpose(tensor, {2, 1, 0});

    EXPECT_EQ(tensor.size(), 12);
    EXPECT_EQ(tensor.shape(), (std::array<size_t, 3>{3, 2, 2}));
    EXPECT_EQ(tensor.strides(), (std::array<size_t, 3>{1, 3, 6}));
    EXPECT_EQ(tensor.start(), 0);
    EXPECT_EQ(tensor.order(), vt::Order::F);
    EXPECT_EQ(tensor.contiguous(), true);
    EXPECT_EQ(vt::asvector(tensor), (std::vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}));

    EXPECT_EQ(tensor1.size(), 12);
    EXPECT_EQ(tensor1.shape(), (std::array<size_t, 3>{2, 2, 3}));
    EXPECT_EQ(tensor1.strides(), (std::array<size_t, 3>{6, 3, 1}));
    EXPECT_EQ(tensor1.start(), 0);
    EXPECT_EQ(tensor1.order(), vt::Order::F);
    EXPECT_EQ(tensor1.contiguous(), false);
    EXPECT_EQ(vt::asvector(tensor), (std::vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}));
    EXPECT_EQ(vt::asvector(tensor1), (std::vector<float>{0, 6, 3, 9, 1, 7, 4, 10, 2, 8, 5, 11}));
}

TEST(MoveAxisC, BasicAssertions) {
    auto tensor = vt::arange(12).reshape(3, 2, 2);
    auto tensor1 = vt::moveaxis(tensor, 0, 2);
    tensor1 = vt::moveaxis(tensor1, 0, 1);

    EXPECT_EQ(tensor.size(), 12);
    EXPECT_EQ(tensor.shape(), (std::array<size_t, 3>{3, 2, 2}));
    EXPECT_EQ(tensor.strides(), (std::array<size_t, 3>{4, 2, 1}));
    EXPECT_EQ(tensor.start(), 0);
    EXPECT_EQ(tensor.order(), vt::Order::C);
    EXPECT_EQ(tensor.contiguous(), true);
    EXPECT_EQ(vt::asvector(tensor), (std::vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}));

    EXPECT_EQ(tensor1.size(), 12);
    EXPECT_EQ(tensor1.shape(), (std::array<size_t, 3>{2, 2, 3}));
    EXPECT_EQ(tensor1.strides(), (std::array<size_t, 3>{1, 2, 4}));
    EXPECT_EQ(tensor1.start(), 0);
    EXPECT_EQ(tensor1.order(), vt::Order::C);
    EXPECT_EQ(tensor1.contiguous(), false);
    EXPECT_EQ(vt::asvector(tensor), (std::vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}));
    EXPECT_EQ(vt::asvector(tensor1), (std::vector<float>{0, 4, 8, 2, 6, 10, 1, 5, 9, 3, 7, 11}));
}

TEST(MoveAxisF, BasicAssertions) {
    auto tensor = vt::arange(12, vt::Order::F).reshape(3, 2, 2);
    auto tensor1 = vt::moveaxis(tensor, 0, 2);
    tensor1 = vt::moveaxis(tensor1, 0, 1);

    EXPECT_EQ(tensor.size(), 12);
    EXPECT_EQ(tensor.shape(), (std::array<size_t, 3>{3, 2, 2}));
    EXPECT_EQ(tensor.strides(), (std::array<size_t, 3>{1, 3, 6}));
    EXPECT_EQ(tensor.start(), 0);
    EXPECT_EQ(tensor.order(), vt::Order::F);
    EXPECT_EQ(tensor.contiguous(), true);
    EXPECT_EQ(vt::asvector(tensor), (std::vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}));

    EXPECT_EQ(tensor1.size(), 12);
    EXPECT_EQ(tensor1.shape(), (std::array<size_t, 3>{2, 2, 3}));
    EXPECT_EQ(tensor1.strides(), (std::array<size_t, 3>{6, 3, 1}));
    EXPECT_EQ(tensor1.start(), 0);
    EXPECT_EQ(tensor1.order(), vt::Order::F);
    EXPECT_EQ(tensor1.contiguous(), false);
    EXPECT_EQ(vt::asvector(tensor), (std::vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}));
    EXPECT_EQ(vt::asvector(tensor1), (std::vector<float>{0, 6, 3, 9, 1, 7, 4, 10, 2, 8, 5, 11}));
}

TEST(SwapaxesC, BasicAssertions) {
    auto tensor = vt::arange(24)({0, 24, 2}).reshape(3, 4);
    auto tensor2 = vt::swapaxes(tensor, 0, 1);
    EXPECT_EQ(vt::asvector(tensor2), (std::vector<float>{0, 8, 16, 2, 10, 18, 4, 12, 20, 6, 14, 22}));
}

TEST(SwapaxesF, BasicAssertions) {
    auto tensor = vt::arange(24, vt::Order::F)({0, 24, 2}).reshape(3, 4);
    auto tensor2 = vt::swapaxes(tensor, 0, 1);
    EXPECT_EQ(vt::asvector(tensor2), (std::vector<float>{0, 6, 12, 18, 2, 8, 14, 20, 4, 10, 16, 22}));
}
