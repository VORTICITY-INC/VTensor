#include <gtest/gtest.h>

#include <lib/vtensor.hpp>

TEST(HelperFunctions, BasicAssertions) {
    auto shape = std::array<size_t, 3>{1, 2, 3};
    auto stides = vt::get_strides(shape);
    auto size = vt::get_size(shape);
    EXPECT_EQ(size, 6);
    EXPECT_EQ(stides[0], 6);
    EXPECT_EQ(stides[1], 3);
    EXPECT_EQ(stides[2], 1);
}

TEST(TensorConstructor, BasicAssertions) {
    using vector_type = vt::Tensor<float, 1>::vector_type;

    auto shape = std::array<size_t, 3>{1, 2, 3};
    auto tensor = vt::Tensor<float, 3>(shape);
    EXPECT_EQ(tensor.size(), 6);
    EXPECT_EQ(tensor.shape(), (std::array<size_t, 3>{1, 2, 3}));
    EXPECT_EQ(tensor.strides(), (std::array<size_t, 3>{6, 3, 1}));
    EXPECT_EQ(tensor.start(), 0);
    EXPECT_EQ(tensor.sliced(), false);
    EXPECT_EQ(vt::asvector(tensor), (std::vector<float>{0, 0, 0, 0, 0, 0}));

    vector_type data(6, 1);
    auto tensor1 = vt::Tensor<float, 3>(std::make_shared<vector_type>(data), shape);
    EXPECT_EQ(tensor1.size(), 6);
    EXPECT_EQ(tensor1.shape(), (std::array<size_t, 3>{1, 2, 3}));
    EXPECT_EQ(tensor1.strides(), (std::array<size_t, 3>{6, 3, 1}));
    EXPECT_EQ(tensor1.start(), 0);
    EXPECT_EQ(tensor1.sliced(), false);
    EXPECT_EQ(vt::asvector(tensor1), (std::vector<float>{1, 1, 1, 1, 1, 1}));

    auto shape2 = std::array<size_t, 1>{5};
    auto strides2 = std::array<size_t, 1>{1};
    auto tensor2 = vt::Tensor<float, 1>(std::make_shared<vector_type>(data), shape2, strides2, 1, true);
    EXPECT_EQ(tensor2.size(), 5);
    EXPECT_EQ(tensor2.shape(), (std::array<size_t, 1>{5}));
    EXPECT_EQ(tensor2.strides(), (std::array<size_t, 1>{1}));
    EXPECT_EQ(tensor2.start(), 1);
    EXPECT_EQ(tensor2.sliced(), true);
    EXPECT_EQ(vt::asvector(tensor2), (std::vector<float>{1, 1, 1, 1, 1}));
}

TEST(TensorReshape, BasicAssertions) {
    auto tensor = vt::arange(6).reshape(1, 2, 3);
    EXPECT_EQ(tensor.size(), 6);
    EXPECT_EQ(tensor.shape(), (std::array<size_t, 3>{1, 2, 3}));
    EXPECT_EQ(tensor.strides(), (std::array<size_t, 3>{6, 3, 1}));
    EXPECT_EQ(tensor.start(), 0);
    EXPECT_EQ(tensor.sliced(), false);
    EXPECT_EQ(vt::asvector(tensor), (std::vector<float>{0, 1, 2, 3, 4, 5}));
}

TEST(SlicedTensorReshape, BasicAssertions) {
    auto tensor = vt::arange(12)({0, 12, 2});
    auto tensor1 = tensor.reshape(1, 2, 3);
    EXPECT_EQ(tensor.size(), 6);
    EXPECT_EQ(tensor1.shape(), (std::array<size_t, 3>{1, 2, 3}));
    EXPECT_EQ(tensor1.strides(), (std::array<size_t, 3>{6, 3, 1}));
    EXPECT_EQ(tensor1.start(), 0);
    EXPECT_EQ(tensor1.sliced(), false);
    EXPECT_EQ(vt::asvector(tensor), (std::vector<float>{0, 2, 4, 6, 8, 10}));
}

TEST(TensorInPlacePlusScalar, BasicAssertions) {
    auto tensor = vt::arange(6);
    tensor += 1;
    EXPECT_EQ(vt::asvector(tensor), (std::vector<float>{1, 2, 3, 4, 5, 6}));
}

TEST(TensorInPlaceMinusScalar, BasicAssertions) {
    auto tensor = vt::arange(6);
    tensor -= 1;
    EXPECT_EQ(vt::asvector(tensor), (std::vector<float>{-1, 0, 1, 2, 3, 4}));
}

TEST(TensorInPlacePlusTensor, BasicAssertions) {
    auto tensor = vt::arange(6);
    auto tensor1 = vt::arange(6);
    tensor += tensor1;
    EXPECT_EQ(vt::asvector(tensor), (std::vector<float>{0, 2, 4, 6, 8, 10}));
    EXPECT_EQ(vt::asvector(tensor1), (std::vector<float>{0, 1, 2, 3, 4, 5}));
}

TEST(TensorInPlaceMinusTensor, BasicAssertions) {
    auto tensor = vt::arange(6);
    auto tensor1 = vt::arange(6);
    tensor -= tensor1;
    EXPECT_EQ(vt::asvector(tensor), (std::vector<float>{0, 0, 0, 0, 0, 0}));
    EXPECT_EQ(vt::asvector(tensor1), (std::vector<float>{0, 1, 2, 3, 4, 5}));
}

TEST(TensorIterator, BasicAssertions) {
    auto tensor = vt::arange(6);
    auto it = tensor.begin();
    auto end = tensor.end();
    EXPECT_EQ(*it, 0);
    ASSERT_NE(it, end);
    size_t index = 0;
    for (; it != end; ++it, ++index) EXPECT_EQ(*it, index);
}

TEST(TensorSliceOperation, BasicAssertions) {
    auto tensor1 = vt::zeros(2);
    tensor1(vt::Slice(0, 2, 1));
    tensor1({0, 2, 1});

    auto tensor2 = vt::zeros(2, 2);
    tensor2(vt::Slice(0, 2, 1), vt::Slice(0, 2, 1));
    tensor2({0, 2, 1}, {0, 2, 1});

    auto tensor3 = vt::zeros(vt::Shape<3>{2, 2, 2});
    tensor3(vt::Slice(0, 2, 1), vt::Slice(0, 2, 1), vt::Slice(0, 2, 1));
    tensor3({0, 2, 1}, {0, 2, 1}, {0, 2, 1});
}

TEST(TensorSliceAlongTheAxisOperation, BasicAssertions) {
    auto tensor = vt::arange(12).reshape(2, 2, 3);
    EXPECT_EQ(vt::asvector(tensor[0]), (std::vector<float>{0, 1, 2, 3, 4, 5}));
    EXPECT_EQ(vt::asvector(tensor[1]), (std::vector<float>{6, 7, 8, 9, 10, 11}));
}

TEST(TensorApplySlices, BasicAssertions) {
    auto tensor1 = vt::arange(12);
    std::array<vt::Slice, 1> slices = {vt::Slice(0, 12, 2)};
    auto tensor2 = tensor1.apply_slices(slices);
    EXPECT_EQ(vt::asvector(tensor2), (std::vector<float>{0, 2, 4, 6, 8, 10}));
    EXPECT_EQ(tensor2.size(), 6);
    EXPECT_EQ(tensor2.shape(), (std::array<size_t, 1>{6}));
    EXPECT_EQ(tensor2.strides(), (std::array<size_t, 1>{2}));
    EXPECT_EQ(tensor2.start(), 0);
    EXPECT_EQ(tensor2.sliced(), true);
}
