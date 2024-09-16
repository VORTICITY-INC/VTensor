#include <gtest/gtest.h>

#include <lib/vtensor.hpp>

TEST(AdditionOperator, BasicAssertions) {
    auto tensor1 = vt::ones(12);
    auto tensor2 = vt::arange(12);
    auto tensor3 = tensor1 + tensor2;
    EXPECT_EQ(vt::asvector(tensor3), (std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}));

    tensor3 = tensor1 + 1.0f;
    EXPECT_EQ(vt::asvector(tensor3), (std::vector<float>{2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2}));

    tensor3 = 1.0f + tensor2;
    EXPECT_EQ(vt::asvector(tensor3), (std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}));

    EXPECT_EQ(vt::asvector(tensor1), (std::vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}));
    EXPECT_EQ(vt::asvector(tensor2), (std::vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}));
}

TEST(SubtractionOperator, BasicAssertions) {
    auto tensor1 = vt::ones(12);
    auto tensor2 = vt::arange(12);
    auto tensor3 = tensor2 - tensor1;
    EXPECT_EQ(vt::asvector(tensor3), (std::vector<float>{-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}));

    tensor3 = tensor1 - 1.0f;
    EXPECT_EQ(vt::asvector(tensor3), (std::vector<float>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));

    tensor3 = 1.0f - tensor1;
    EXPECT_EQ(vt::asvector(tensor3), (std::vector<float>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));

    EXPECT_EQ(vt::asvector(tensor1), (std::vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}));
    EXPECT_EQ(vt::asvector(tensor2), (std::vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}));
}

TEST(AdditionOperatorForReshapeAndSliceTensor, BasicAssertions) {
    auto tensor1 = vt::ones(12).reshape(4, 3);
    auto tensor2 = vt::arange(12).reshape(4, 3);
    auto tensor3 = tensor1({1, 3, 1}, {0, 3, 2});
    auto tensor4 = tensor2({1, 3, 1}, {0, 3, 2});
    auto tensor5 = tensor3 + tensor4;
    EXPECT_EQ(vt::asvector(tensor5), (std::vector<float>{4, 6, 7, 9}));

    tensor5 = tensor3 + 1.0f;
    EXPECT_EQ(vt::asvector(tensor5), (std::vector<float>{2, 2, 2, 2}));

    tensor5 = 1.0f + tensor4;
    EXPECT_EQ(vt::asvector(tensor5), (std::vector<float>{4, 6, 7, 9}));

    EXPECT_EQ(vt::asvector(tensor1), (std::vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}));
    EXPECT_EQ(vt::asvector(tensor2), (std::vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}));
}

TEST(SubtractionOperatorForReshapeAndSliceTensor, BasicAssertions) {
    auto tensor1 = vt::ones(12).reshape(4, 3);
    auto tensor2 = vt::arange(12).reshape(4, 3);
    auto tensor3 = tensor1({1, 3, 1}, {0, 3, 2});
    auto tensor4 = tensor2({1, 3, 1}, {0, 3, 2});
    auto tensor5 = tensor4 - tensor3;
    EXPECT_EQ(vt::asvector(tensor5), (std::vector<float>{2, 4, 5, 7}));

    tensor5 = tensor3 - 1.0f;
    EXPECT_EQ(vt::asvector(tensor5), (std::vector<float>{0, 0, 0, 0}));

    tensor5 = 1.0f - tensor3;
    EXPECT_EQ(vt::asvector(tensor5), (std::vector<float>{0, 0, 0, 0}));

    EXPECT_EQ(vt::asvector(tensor1), (std::vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}));
    EXPECT_EQ(vt::asvector(tensor2), (std::vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}));
}

TEST(OperatorMultiply, BasicAssertions) {
    auto tensor1 = vt::ones(12).reshape(4, 3) + 5.0f;
    auto tensor2 = vt::arange(12).reshape(4, 3);
    auto tensor3 = tensor1({1, 3, 1}, {0, 3, 2});
    auto tensor4 = tensor2({1, 3, 1}, {0, 3, 2});
    auto tensor5 = tensor4 * tensor3;
    EXPECT_EQ(vt::asvector(tensor5), (std::vector<float>{18, 30, 36, 48}));

    tensor5 = tensor4 * 6.0f;
    EXPECT_EQ(vt::asvector(tensor5), (std::vector<float>{18, 30, 36, 48}));

    tensor5 = 6.0f * tensor4;
    EXPECT_EQ(vt::asvector(tensor5), (std::vector<float>{18, 30, 36, 48}));
}

TEST(OperatorDevide, BasicAssertions) {
    auto tensor1 = vt::ones(12).reshape(4, 3) + 5.0f;
    auto tensor2 = vt::arange(12).reshape(4, 3);
    auto tensor3 = tensor1({1, 3, 1}, {0, 3, 2});
    auto tensor4 = tensor2({1, 3, 1}, {0, 3, 2});
    auto tensor5 = tensor4 / tensor3;

    auto vec = std::vector<float>{3, 5, 6, 8};
    auto result = vt::asvector(tensor5);
    for (size_t i = 0; i < vec.size(); i++) {
        EXPECT_NEAR(result[i], vec[i] / 6.0f, 1e-6);
    }

    tensor5 = tensor4 / 6.0f;
    result = vt::asvector(tensor5);
    for (size_t i = 0; i < vec.size(); i++) {
        EXPECT_NEAR(result[i], vec[i] / 6.0f, 1e-6);
    }

    tensor5 = 6.0f / tensor4;
    result = vt::asvector(tensor5);
    for (size_t i = 0; i < vec.size(); i++) {
        EXPECT_NEAR(result[i], 6.0f / vec[i], 1e-6);
    }
}

TEST(OperatorGreaterThan, BasicAssertions) {
    auto tensor1 = vt::ones(12).reshape(4, 3) + 5.0f;
    auto tensor2 = vt::arange(12).reshape(4, 3);
    auto tensor3 = tensor1({1, 3, 1}, {0, 3, 2});
    auto tensor4 = tensor2({1, 3, 1}, {0, 3, 2});
    auto tensor5 = tensor4 > tensor3;
    EXPECT_EQ(vt::asvector(tensor5), (std::vector<bool>{0, 0, 0, 1}));

    tensor5 = tensor4 > 5.0f;
    EXPECT_EQ(vt::asvector(tensor5), (std::vector<bool>{0, 0, 1, 1}));

    tensor5 = 5.0f > tensor4;
    EXPECT_EQ(vt::asvector(tensor5), (std::vector<bool>{1, 0, 0, 0}));
}

TEST(OperatorLessThan, BasicAssertions) {
    auto tensor1 = vt::ones(12).reshape(4, 3) + 5.0f;
    auto tensor2 = vt::arange(12).reshape(4, 3);
    auto tensor3 = tensor1({1, 3, 1}, {0, 3, 2});
    auto tensor4 = tensor2({1, 3, 1}, {0, 3, 2});
    auto tensor5 = tensor4 < tensor3;
    EXPECT_EQ(vt::asvector(tensor5), (std::vector<bool>{1, 1, 0, 0}));

    tensor5 = tensor4 < 5.0f;
    EXPECT_EQ(vt::asvector(tensor5), (std::vector<bool>{1, 0, 0, 0}));

    tensor5 = 5.0f < tensor4;
    EXPECT_EQ(vt::asvector(tensor5), (std::vector<bool>{0, 0, 1, 1}));
}

TEST(OperatorGreaterThanOrEqual, BasicAssertions) {
    auto tensor1 = vt::ones(12).reshape(4, 3) + 5.0f;
    auto tensor2 = vt::arange(12).reshape(4, 3);
    auto tensor3 = tensor1({1, 3, 1}, {0, 3, 2});
    auto tensor4 = tensor2({1, 3, 1}, {0, 3, 2});
    auto tensor5 = tensor4 >= tensor3;
    EXPECT_EQ(vt::asvector(tensor5), (std::vector<bool>{0, 0, 1, 1}));

    tensor5 = tensor4 >= 5.0f;
    EXPECT_EQ(vt::asvector(tensor5), (std::vector<bool>{0, 1, 1, 1}));

    tensor5 = 5.0f >= tensor4;
    EXPECT_EQ(vt::asvector(tensor5), (std::vector<bool>{1, 1, 0, 0}));
}

TEST(OperatorLessThanOrEqual, BasicAssertions) {
    auto tensor1 = vt::ones(12).reshape(4, 3) + 5.0f;
    auto tensor2 = vt::arange(12).reshape(4, 3);
    auto tensor3 = tensor1({1, 3, 1}, {0, 3, 2});
    auto tensor4 = tensor2({1, 3, 1}, {0, 3, 2});
    auto tensor5 = tensor4 <= tensor3;
    EXPECT_EQ(vt::asvector(tensor5), (std::vector<bool>{1, 1, 1, 0}));

    tensor5 = tensor4 <= 5.0f;
    EXPECT_EQ(vt::asvector(tensor5), (std::vector<bool>{1, 1, 0, 0}));

    tensor5 = 5.0f <= tensor4;
    EXPECT_EQ(vt::asvector(tensor5), (std::vector<bool>{0, 1, 1, 1}));
}

TEST(OperatorEqual, BasicAssertions) {
    auto tensor1 = vt::ones(12).reshape(4, 3) + 5.0f;
    auto tensor2 = vt::arange(12).reshape(4, 3);
    auto tensor3 = tensor1({1, 3, 1}, {0, 3, 2});
    auto tensor4 = tensor2({1, 3, 1}, {0, 3, 2});
    auto tensor5 = tensor4 == tensor3;
    EXPECT_EQ(vt::asvector(tensor5), (std::vector<bool>{0, 0, 1, 0}));

    tensor5 = tensor4 == 5.0f;
    EXPECT_EQ(vt::asvector(tensor5), (std::vector<bool>{0, 1, 0, 0}));

    tensor5 = 5.0f == tensor4;
    EXPECT_EQ(vt::asvector(tensor5), (std::vector<bool>{0, 1, 0, 0}));
}

TEST(OperatorNotEqual, BasicAssertions) {
    auto tensor1 = vt::ones(12).reshape(4, 3) + 5.0f;
    auto tensor2 = vt::arange(12).reshape(4, 3);
    auto tensor3 = tensor1({1, 3, 1}, {0, 3, 2});
    auto tensor4 = tensor2({1, 3, 1}, {0, 3, 2});
    auto tensor5 = tensor4 != tensor3;
    EXPECT_EQ(vt::asvector(tensor5), (std::vector<bool>{1, 1, 0, 1}));

    tensor5 = tensor4 != 5.0f;
    EXPECT_EQ(vt::asvector(tensor5), (std::vector<bool>{1, 0, 1, 1}));

    tensor5 = 5.0f != tensor4;
    EXPECT_EQ(vt::asvector(tensor5), (std::vector<bool>{1, 0, 1, 1}));
}