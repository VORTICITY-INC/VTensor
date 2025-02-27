#include <gtest/gtest.h>

#include <lib/vtensor.hpp>

TEST(SliceConstructor, BasicAssertions) {
    vt::Slice s{1};
    EXPECT_EQ(s.start, 1);
    EXPECT_EQ(s.end, 2);
    EXPECT_EQ(s.step, 1);

    s = {1, 10};
    EXPECT_EQ(s.start, 1);
    EXPECT_EQ(s.end, 10);
    EXPECT_EQ(s.step, 1);

    s = {1, 10, 2};
    EXPECT_EQ(s.start, 1);
    EXPECT_EQ(s.end, 10);
    EXPECT_EQ(s.step, 2);
}

TEST(Slice1DTensorC, BasicAssertions) {
    auto tensor = vt::arange(12);
    auto tensor1 = tensor({1, 10, 2});
    EXPECT_EQ(vt::asvector(tensor), (std::vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}));
    EXPECT_EQ(vt::asvector(tensor1), (std::vector<float>{1, 3, 5, 7, 9}));
    EXPECT_EQ(tensor.contiguous(), true);
    EXPECT_EQ(tensor1.contiguous(), false);
    EXPECT_EQ(tensor1.order(), vt::Order::C);
    EXPECT_EQ(tensor1.size(), 5);
}

TEST(Slice1DTensorF, BasicAssertions) {
    auto tensor = vt::arange(12, vt::Order::F);
    auto tensor1 = tensor({1, 10, 2});
    EXPECT_EQ(vt::asvector(tensor), (std::vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}));
    EXPECT_EQ(vt::asvector(tensor1), (std::vector<float>{1, 3, 5, 7, 9}));
    EXPECT_EQ(tensor.contiguous(), true);
    EXPECT_EQ(tensor1.contiguous(), false);
    EXPECT_EQ(tensor1.order(), vt::Order::F);
    EXPECT_EQ(tensor1.size(), 5);
}

TEST(Slice2DTensorC, BasicAssertions) {
    auto tensor = vt::arange(12).reshape(4, 3);
    auto tensor1 = tensor({1, 3, 1}, {0, 3, 2});
    EXPECT_EQ(vt::asvector(tensor), (std::vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}));
    EXPECT_EQ(vt::asvector(tensor1), (std::vector<float>{3, 5, 6, 8}));
    EXPECT_EQ(tensor.contiguous(), true);
    EXPECT_EQ(tensor1.contiguous(), false);
    EXPECT_EQ(tensor1.order(), vt::Order::C);
    EXPECT_EQ(tensor1.size(), 4);
}

TEST(Slice2DTensorF, BasicAssertions) {
    auto tensor = vt::arange(12, vt::Order::F).reshape(4, 3);
    auto tensor1 = tensor({1, 3, 1}, {0, 3, 2});
    EXPECT_EQ(vt::asvector(tensor), (std::vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}));
    EXPECT_EQ(vt::asvector(tensor1), (std::vector<float>{1, 2, 9, 10}));
    EXPECT_EQ(tensor.contiguous(), true);
    EXPECT_EQ(tensor1.contiguous(), false);
    EXPECT_EQ(tensor1.order(), vt::Order::F);
    EXPECT_EQ(tensor1.size(), 4);
}

TEST(SliceAndAssigmentFromTensorC, BasicAssertions) {
    auto tensor1 = vt::ones(4, 3);
    auto tensor2 = vt::arange(12).reshape(4, 3);
    vt::Tensor tensor3 = tensor2({1, 3, 1}, {0, 3, 2});
    tensor1({1, 3, 1}, {0, 3, 2}) = tensor3;
    EXPECT_EQ(vt::asvector(tensor1), (std::vector<float>{1, 1, 1, 3, 1, 5, 6, 1, 8, 1, 1, 1}));
    EXPECT_EQ(vt::asvector(tensor2), (std::vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}));
    EXPECT_EQ(vt::asvector(tensor3), (std::vector<float>{3, 5, 6, 8}));
    EXPECT_EQ(tensor1.contiguous(), true);
    EXPECT_EQ(tensor2.contiguous(), true);
    EXPECT_EQ(tensor3.contiguous(), false);
    EXPECT_EQ(tensor3.size(), 4);
}

TEST(SliceAndAssigmentFromTensorF, BasicAssertions) {
    auto tensor1 = vt::ones(4, 3, vt::Order::F);
    auto tensor2 = vt::arange(12, vt::Order::F).reshape(4, 3);
    vt::Tensor tensor3 = tensor2({1, 3, 1}, {0, 3, 2});
    tensor1({1, 3, 1}, {0, 3, 2}) = tensor3;
    EXPECT_EQ(vt::asvector(tensor1), (std::vector<float>{1, 1, 2, 1, 1, 1, 1, 1, 1, 9, 10, 1}));
    EXPECT_EQ(vt::asvector(tensor2), (std::vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}));
    EXPECT_EQ(vt::asvector(tensor3), (std::vector<float>{1, 2, 9, 10}));
    EXPECT_EQ(tensor1.contiguous(), true);
    EXPECT_EQ(tensor2.contiguous(), true);
    EXPECT_EQ(tensor3.contiguous(), false);
    EXPECT_EQ(tensor3.size(), 4);
}

TEST(SliceAndAssigmentFromProxyC, BasicAssertions) {
    auto tensor1 = vt::ones(4, 3);
    auto tensor2 = vt::arange(12).reshape(4, 3);
    tensor1({1, 3, 1}, {0, 3, 2}) = tensor2({1, 3, 1}, {0, 3, 2});
    EXPECT_EQ(vt::asvector(tensor1), (std::vector<float>{1, 1, 1, 3, 1, 5, 6, 1, 8, 1, 1, 1}));
    EXPECT_EQ(vt::asvector(tensor2), (std::vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}));
    EXPECT_EQ(tensor1.contiguous(), true);
    EXPECT_EQ(tensor2.contiguous(), true);
}

TEST(SliceAndAssigmentFromProxyF, BasicAssertions) {
    auto tensor1 = vt::ones(4, 3, vt::Order::F);
    auto tensor2 = vt::arange(12, vt::Order::F).reshape(4, 3);
    tensor1({1, 3, 1}, {0, 3, 2}) = tensor2({1, 3, 1}, {0, 3, 2});
    EXPECT_EQ(vt::asvector(tensor1), (std::vector<float>{1, 1, 2, 1, 1, 1, 1, 1, 1, 9, 10, 1}));
    EXPECT_EQ(vt::asvector(tensor2), (std::vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}));
    EXPECT_EQ(tensor1.contiguous(), true);
    EXPECT_EQ(tensor2.contiguous(), true);
}

TEST(SliceAndAssigmentFromConstantC, BasicAssertions) {
    auto tensor1 = vt::ones(4, 3);
    tensor1({1, 3, 1}, {0, 3, 2}) = 2;
    EXPECT_EQ(vt::asvector(tensor1), (std::vector<float>{1, 1, 1, 2, 1, 2, 2, 1, 2, 1, 1, 1}));
    EXPECT_EQ(tensor1.contiguous(), true);
}

TEST(SliceAndAssigmentFromConstantF, BasicAssertions) {
    auto tensor1 = vt::ones(4, 3, vt::Order::F);
    tensor1({1, 3, 1}, {0, 3, 2}) = 2;
    EXPECT_EQ(vt::asvector(tensor1), (std::vector<float>{1, 2, 2, 1, 1, 1, 1, 1, 1, 2, 2, 1}));
    EXPECT_EQ(tensor1.contiguous(), true);
}

TEST(TensorCondProxyFromConstantC, BasicAssertions) {
    auto tensor = vt::arange(24)({0, 24, 2}).reshape(3, 4);
    tensor[tensor > 12.0f] = 1.0f;
    EXPECT_EQ(vt::asvector(tensor), (std::vector<float>{0, 2, 4, 6, 8, 10, 12, 1, 1, 1, 1, 1}));
}

TEST(TensorCondProxyFromConstantF, BasicAssertions) {
    auto tensor = vt::arange(24, vt::Order::F)({0, 24, 2}).reshape(3, 4);
    tensor[tensor > 12.0f] = 1.0f;
    EXPECT_EQ(vt::asvector(tensor), (std::vector<float>{0, 2, 4, 6, 8, 10, 12, 1, 1, 1, 1, 1}));
}

TEST(BroadcastSliceAssignmentC, BasicAssertions) {
    auto tensor = vt::arange(12).reshape(4, 3);
    tensor(vt::Slice::all(), vt::Slice(0, 2, 1)) = tensor(vt::Slice::all(), vt::Slice(2, 3, 1));
    EXPECT_EQ(vt::asvector(tensor), (std::vector<float>{2, 2, 2, 5, 5, 5, 8, 8, 8, 11, 11, 11}));
    EXPECT_EQ(tensor.contiguous(), true);
}

TEST(BroadcastSliceAssignmentF, BasicAssertions) {
    auto tensor = vt::arange(12, vt::Order::F).reshape(4, 3);
    tensor(vt::Slice::all(), vt::Slice(0, 2, 1)) = tensor(vt::Slice::all(), vt::Slice(2, 3, 1));
    EXPECT_EQ(vt::asvector(tensor), (std::vector<float>{8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11}));
    EXPECT_EQ(tensor.contiguous(), true);
}
