#include <gtest/gtest.h>

#include <lib/vtensor.hpp>

TEST(HelperFunctions, BasicAssertions) {
    auto shape = vt::Shape<3>{1, 2, 3};
    auto stides = vt::get_strides(shape);
    auto size = vt::get_size(shape);
    auto expanded_shape = vt::expand_shape(shape, 4, 5, 6);
    EXPECT_EQ(size, 6);
    EXPECT_EQ(stides[0], 6);
    EXPECT_EQ(stides[1], 3);
    EXPECT_EQ(stides[2], 1);
    for (size_t i = 0; i < 6; i++) {
        EXPECT_EQ(expanded_shape[i], i + 1);
    }

    // Test for 0D tensor.
    auto shape1 = vt::Shape<0>{};
    auto stides1 = vt::get_strides(shape1);
    auto size1 = vt::get_size(shape1);
    auto expanded_shape1 = vt::expand_shape(shape1, 1, 2, 3);
    EXPECT_EQ(size1, 1);
    for (size_t i = 0; i < 3; i++) {
        EXPECT_EQ(expanded_shape1[i], i + 1);
    }
}

TEST(TensorConstructor, BasicAssertions) {
    using vector_type = vt::Tensor<float, 1>::vector_type;

    auto shape = vt::Shape<3>{1, 2, 3};
    auto tensor = vt::Tensor<float, 3>(shape);
    EXPECT_EQ(tensor.size(), 6);
    EXPECT_EQ(tensor.shape(), (vt::Shape<3>{1, 2, 3}));
    EXPECT_EQ(tensor.strides(), (vt::Shape<3>{6, 3, 1}));
    EXPECT_EQ(tensor.start(), 0);
    EXPECT_EQ(tensor.contiguous(), true);
    EXPECT_EQ(vt::asvector(tensor), (std::vector<float>{0, 0, 0, 0, 0, 0}));

    vector_type data(6, 1);
    auto tensor1 = vt::Tensor<float, 3>(std::make_shared<vector_type>(data), shape);
    EXPECT_EQ(tensor1.size(), 6);
    EXPECT_EQ(tensor1.shape(), (vt::Shape<3>{1, 2, 3}));
    EXPECT_EQ(tensor1.strides(), (vt::Shape<3>{6, 3, 1}));
    EXPECT_EQ(tensor1.start(), 0);
    EXPECT_EQ(tensor1.contiguous(), true);
    EXPECT_EQ(vt::asvector(tensor1), (std::vector<float>{1, 1, 1, 1, 1, 1}));

    auto shape2 = vt::Shape<1>{5};
    auto strides2 = vt::Shape<1>{1};
    auto tensor2 = vt::Tensor<float, 1>(std::make_shared<vector_type>(data), shape2, strides2, 1, true);
    EXPECT_EQ(tensor2.size(), 5);
    EXPECT_EQ(tensor2.shape(), (vt::Shape<1>{5}));
    EXPECT_EQ(tensor2.strides(), (vt::Shape<1>{1}));
    EXPECT_EQ(tensor2.start(), 1);
    EXPECT_EQ(tensor2.contiguous(), true);
    EXPECT_EQ(vt::asvector(tensor2), (std::vector<float>{1, 1, 1, 1, 1}));

    // Test for 0D tensor.
    auto tensor3 = vt::Tensor<float, 0>(vt::Shape<0>{});
    EXPECT_EQ(tensor3.size(), 1);
    EXPECT_EQ(tensor3.shape(), (vt::Shape<0>{}));
    EXPECT_EQ(tensor3.strides(), (vt::Shape<0>{}));
    EXPECT_EQ(tensor3.start(), 0);
    EXPECT_EQ(tensor3.contiguous(), true);
    EXPECT_EQ(vt::asvector(tensor3), (std::vector<float>{0}));
}

TEST(TensorReshape, BasicAssertions) {
    auto tensor = vt::arange(6).reshape(1, 2, 3);
    EXPECT_EQ(tensor.size(), 6);
    EXPECT_EQ(tensor.shape(), (vt::Shape<3>{1, 2, 3}));
    EXPECT_EQ(tensor.strides(), (vt::Shape<3>{6, 3, 1}));
    EXPECT_EQ(tensor.start(), 0);
    EXPECT_EQ(tensor.contiguous(), true);
    EXPECT_EQ(vt::asvector(tensor), (std::vector<float>{0, 1, 2, 3, 4, 5}));

    // Test for 0D tensor.
    auto tensor1 = vt::Tensor<float, 0>(vt::Shape<0>{}).reshape(1, 1);
    EXPECT_EQ(tensor1.size(), 1);
    EXPECT_EQ(tensor1.shape(), (vt::Shape<2>{1, 1}));
    EXPECT_EQ(tensor1.strides(), (vt::Shape<2>{1, 1}));
    EXPECT_EQ(tensor1.start(), 0);
    EXPECT_EQ(tensor1.contiguous(), true);

}

TEST(SlicedTensorReshape, BasicAssertions) {
    auto tensor = vt::arange(12)({0, 12, 2});
    auto tensor1 = tensor.reshape(1, 2, 3);
    EXPECT_EQ(tensor.size(), 6);
    EXPECT_EQ(tensor1.shape(), (vt::Shape<3>{1, 2, 3}));
    EXPECT_EQ(tensor1.strides(), (vt::Shape<3>{6, 3, 1}));
    EXPECT_EQ(tensor1.start(), 0);
    EXPECT_EQ(tensor1.contiguous(), true);
    EXPECT_EQ(vt::asvector(tensor), (std::vector<float>{0, 2, 4, 6, 8, 10}));
}

TEST(TensorInPlacePlusScalar, BasicAssertions) {
    auto tensor = vt::arange(6);
    tensor += 1;
    EXPECT_EQ(vt::asvector(tensor), (std::vector<float>{1, 2, 3, 4, 5, 6}));

    // Test for 0D tensor.
    auto tensor1 = tensor[1];
    tensor1 += 1;
    EXPECT_EQ(vt::asvector(tensor1), (std::vector<float>{3}));
}

TEST(TensorInPlaceMinusScalar, BasicAssertions) {
    auto tensor = vt::arange(6);
    tensor -= 1;
    EXPECT_EQ(vt::asvector(tensor), (std::vector<float>{-1, 0, 1, 2, 3, 4}));

    // Test for 0D tensor.
    auto tensor1 = tensor[1];
    tensor1 -= 1;
    EXPECT_EQ(vt::asvector(tensor1), (std::vector<float>{-1}));
}

TEST(TensorInPlaceMultiplyScalar, BasicAssertions) {
    auto tensor = vt::arange(6);
    tensor *= 2.0f;
    EXPECT_EQ(vt::asvector(tensor), (std::vector<float>{0, 2, 4, 6, 8, 10}));

    // Test for 0D tensor.
    auto tensor1 = tensor[1];
    tensor1 *= 2.0f;
    EXPECT_EQ(vt::asvector(tensor1), (std::vector<float>{4}));
}

TEST(TensorInPlaceDivideScalar, BasicAssertions) {
    auto tensor = vt::arange(6) * 2.0f;
    tensor /= 2.0f;
    EXPECT_EQ(vt::asvector(tensor), (std::vector<float>{0, 1, 2, 3, 4, 5}));

    // Test for 0D tensor.
    auto tensor1 = tensor[1];
    tensor1 /= 2.0f;
    EXPECT_EQ(vt::asvector(tensor1), (std::vector<float>{0.5}));
}

TEST(TensorInPlacePlusTensor, BasicAssertions) {
    auto tensor = vt::arange(6);
    auto tensor1 = vt::arange(6);
    tensor += tensor1;
    EXPECT_EQ(vt::asvector(tensor), (std::vector<float>{0, 2, 4, 6, 8, 10}));
    EXPECT_EQ(vt::asvector(tensor1), (std::vector<float>{0, 1, 2, 3, 4, 5}));

    auto tensor2 = tensor[1] + tensor1[1];
    EXPECT_EQ(vt::asvector(tensor2), (std::vector<float>{3}));
}

TEST(TensorInPlaceMinusTensor, BasicAssertions) {
    auto tensor = vt::arange(6);
    auto tensor1 = vt::arange(6);
    tensor -= tensor1;
    EXPECT_EQ(vt::asvector(tensor), (std::vector<float>{0, 0, 0, 0, 0, 0}));
    EXPECT_EQ(vt::asvector(tensor1), (std::vector<float>{0, 1, 2, 3, 4, 5}));

    auto tensor2 = tensor[1] - tensor1[1];
    EXPECT_EQ(vt::asvector(tensor2), (std::vector<float>{-1}));
}

TEST(TensorInPlaceMultiplyTensor, BasicAssertions) {
    auto tensor = vt::arange(6);
    auto tensor1 = vt::arange(6);
    tensor *= tensor1;
    EXPECT_EQ(vt::asvector(tensor), (std::vector<float>{0, 1, 4, 9, 16, 25}));
    EXPECT_EQ(vt::asvector(tensor1), (std::vector<float>{0, 1, 2, 3, 4, 5}));

    auto tensor2 = tensor[1] * tensor1[1];
    EXPECT_EQ(vt::asvector(tensor2), (std::vector<float>{1}));
}

TEST(TensorInPlaceDivideTensor, BasicAssertions) {
    auto tensor = (vt::arange(6) + 1.0f) * 2.0f;
    auto tensor1 = vt::arange(6) + 1.0f;
    tensor /= tensor1;
    EXPECT_EQ(vt::asvector(tensor), (std::vector<float>{2, 2, 2, 2, 2, 2}));
    EXPECT_EQ(vt::asvector(tensor1), (std::vector<float>{1, 2, 3, 4, 5, 6}));
    auto tensor2 = tensor[1] / tensor1[1];
    EXPECT_EQ(vt::asvector(tensor2), (std::vector<float>{1}));
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

    auto tensor4 = vt::zeros(vt::Shape<3>{2, 2, 2});
    std::array<vt::Slice, 3> slices = {vt::Slice(0, 2, 1), vt::Slice(0, 2, 1), vt::Slice(0, 2, 1)};
    tensor4(slices);
}

TEST(TensorEllipsisSlices, BasicAssertions){
    auto tensor = vt::arange(12).reshape(2, 2, 3);
    tensor = tensor(vt::ellipsis, {0, 2});
    EXPECT_EQ(vt::asvector(tensor), (std::vector<float>{0, 1, 3, 4, 6, 7, 9, 10}));

    auto tensor1 = vt::arange(12);
    tensor1 = tensor1(vt::ellipsis, {0, 2});
    EXPECT_EQ(vt::asvector(tensor1), (std::vector<float>{0, 1}));

}

TEST(TensorSliceAlongTheAxisOperation, BasicAssertions) {
    auto tensor = vt::arange(12).reshape(2, 2, 3);
    EXPECT_EQ(vt::asvector(tensor[0]), (std::vector<float>{0, 1, 2, 3, 4, 5}));
    EXPECT_EQ(vt::asvector(tensor[1]), (std::vector<float>{6, 7, 8, 9, 10, 11}));

    auto tensor1 = vt::arange(12);
    EXPECT_EQ(vt::asvector(tensor1[5]), (std::vector<float>{5}));
}

TEST(TensorApplySlices, BasicAssertions) {
    auto tensor1 = vt::arange(12);
    std::array<vt::Slice, 1> slices = {vt::Slice(0, 12, 2)};
    auto tensor2 = tensor1.apply_slices(slices);
    EXPECT_EQ(vt::asvector(tensor2), (std::vector<float>{0, 2, 4, 6, 8, 10}));
    EXPECT_EQ(tensor2.size(), 6);
    EXPECT_EQ(tensor2.shape(), (vt::Shape<1>{6}));
    EXPECT_EQ(tensor2.strides(), (vt::Shape<1>{2}));
    EXPECT_EQ(tensor2.start(), 0);
    EXPECT_EQ(tensor2.contiguous(), false);
}

TEST(TensorNewAxisAtFirstAxis, BasicAssertions){
    auto tensor = vt::arange(24)({0, 24, 2}).reshape(3, 4);
    auto tensor1 = tensor(vt::newaxis, vt::ellipsis);
    EXPECT_EQ(vt::asvector(tensor1), (std::vector<float>{0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22}));
    EXPECT_EQ(tensor1.size(), 12);
    EXPECT_EQ(tensor1.shape(), (vt::Shape<3>{1, 3, 4}));
    EXPECT_EQ(tensor1.strides(), (vt::Shape<3>{0, 4, 1}));
    EXPECT_EQ(tensor1.start(), 0);
    EXPECT_EQ(tensor1.contiguous(), false);

    // Test for 0D tensor.
    auto tensor2 = vt::Tensor<float, 0>(vt::Shape<0>{});
    auto tensor3 = tensor2(vt::newaxis, vt::ellipsis);
    EXPECT_EQ(vt::asvector(tensor3), (std::vector<float>{0}));
    EXPECT_EQ(tensor3.size(), 1);
    EXPECT_EQ(tensor3.shape(), (vt::Shape<1>{1}));
    EXPECT_EQ(tensor3.strides(), (vt::Shape<1>{0}));
    EXPECT_EQ(tensor3.start(), 0);
    EXPECT_EQ(tensor3.contiguous(), false);
}

TEST(TensorNewAxisAtLastAxis, BasicAssertions){
    auto tensor = vt::arange(24)({0, 24, 2}).reshape(3, 4);
    auto tensor1 = tensor(vt::ellipsis, vt::newaxis);
    EXPECT_EQ(vt::asvector(tensor1), (std::vector<float>{0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22}));
    EXPECT_EQ(tensor1.size(), 12);
    EXPECT_EQ(tensor1.shape(), (vt::Shape<3>{3, 4, 1}));
    EXPECT_EQ(tensor1.strides(), (vt::Shape<3>{4, 1, 0}));
    EXPECT_EQ(tensor1.start(), 0);
    EXPECT_EQ(tensor1.contiguous(), false);

    // Test for 0D tensor.
    auto tensor2 = vt::Tensor<float, 0>(vt::Shape<0>{});
    auto tensor3 = tensor2(vt::ellipsis, vt::newaxis);
    EXPECT_EQ(vt::asvector(tensor3), (std::vector<float>{0}));
    EXPECT_EQ(tensor3.size(), 1);
    EXPECT_EQ(tensor3.shape(), (vt::Shape<1>{1}));
    EXPECT_EQ(tensor3.strides(), (vt::Shape<1>{0}));
    EXPECT_EQ(tensor3.start(), 0);
    EXPECT_EQ(tensor3.contiguous(), false);
}

TEST(TensorIndexWithCond, BasicAssertions){
    auto tensor = vt::arange(24)({0, 24, 2}).reshape(3, 4);
    tensor[tensor > 12.0f] = 1.0f;
    EXPECT_EQ(vt::asvector(tensor), (std::vector<float>{0, 2, 4, 6, 8, 10, 12, 1, 1, 1, 1, 1}));
}

TEST(TensorAstype, BasicAssertions){
    auto tensor = vt::arange<int>(3);
    auto tensor1 = tensor.astype<float>();
    EXPECT_EQ(vt::asvector(tensor1), (std::vector<float>{0, 1, 2}));
}
