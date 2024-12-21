#include <gtest/gtest.h>

#include <lib/vtensor.hpp>

TEST(AdditionOperatorC, BasicAssertions) {
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

    auto tensor4 = tensor1[1] + tensor2[1];
    EXPECT_EQ(vt::asvector(tensor4), (std::vector<float>{2}));

    auto tensor5 = vt::arange(12)({0, 12, 2}).reshape(1, 2, 3);
    auto tensor6 = vt::arange(2).reshape(2, 1, 1);
    auto tensor7 = tensor5 + tensor6;
    EXPECT_EQ(vt::asvector(tensor7), (std::vector<float>{0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11}));
}

TEST(AdditionOperatorF, BasicAssertions) {
    auto tensor1 = vt::ones(12, vt::Order::F);
    auto tensor2 = vt::arange(12, vt::Order::F);
    auto tensor3 = tensor1 + tensor2;
    EXPECT_EQ(vt::asvector(tensor3), (std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}));

    tensor3 = tensor1 + 1.0f;
    EXPECT_EQ(vt::asvector(tensor3), (std::vector<float>{2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2}));

    tensor3 = 1.0f + tensor2;
    EXPECT_EQ(vt::asvector(tensor3), (std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}));

    EXPECT_EQ(vt::asvector(tensor1), (std::vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}));
    EXPECT_EQ(vt::asvector(tensor2), (std::vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}));

    auto tensor4 = tensor1[1] + tensor2[1];
    EXPECT_EQ(vt::asvector(tensor4), (std::vector<float>{2}));

    auto tensor5 = vt::arange(12, vt::Order::F)({0, 12, 2}).reshape(1, 2, 3);
    auto tensor6 = vt::arange(2, vt::Order::F).reshape(2, 1, 1);
    auto tensor7 = tensor5 + tensor6;    
    EXPECT_EQ(vt::asvector(tensor7), (std::vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}));
}

TEST(SubtractionOperatorC, BasicAssertions) {
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

    auto tensor4 = tensor1[1] - tensor2[1];
    EXPECT_EQ(vt::asvector(tensor4), (std::vector<float>{0}));    

    auto tensor5 = vt::arange(12)({0, 12, 2}).reshape(1, 2, 3);
    auto tensor6 = vt::arange(2).reshape(2, 1, 1);
    auto tensor7 = tensor5 - tensor6;
    EXPECT_EQ(vt::asvector(tensor7), (std::vector<float>{0, 2, 4, 6, 8, 10, -1, 1, 3, 5, 7, 9}));
}

TEST(SubtractionOperatorF, BasicAssertions) {
    auto tensor1 = vt::ones(12, vt::Order::F);
    auto tensor2 = vt::arange(12, vt::Order::F);
    auto tensor3 = tensor2 - tensor1;
    EXPECT_EQ(vt::asvector(tensor3), (std::vector<float>{-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}));

    tensor3 = tensor1 - 1.0f;
    EXPECT_EQ(vt::asvector(tensor3), (std::vector<float>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));

    tensor3 = 1.0f - tensor1;
    EXPECT_EQ(vt::asvector(tensor3), (std::vector<float>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));

    EXPECT_EQ(vt::asvector(tensor1), (std::vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}));
    EXPECT_EQ(vt::asvector(tensor2), (std::vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}));

    auto tensor4 = tensor1[1] - tensor2[1];
    EXPECT_EQ(vt::asvector(tensor4), (std::vector<float>{0}));    

    auto tensor5 = vt::arange(12, vt::Order::F)({0, 12, 2}).reshape(1, 2, 3);
    auto tensor6 = vt::arange(2, vt::Order::F).reshape(2, 1, 1);
    auto tensor7 = tensor5 - tensor6;
    EXPECT_EQ(vt::asvector(tensor7), (std::vector<float>{0, -1, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9}));
}

TEST(AdditionOperatorForReshapeAndSliceTensorC, BasicAssertions) {
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

TEST(AdditionOperatorForReshapeAndSliceTensorF, BasicAssertions) {
    auto tensor1 = vt::ones(12, vt::Order::F).reshape(4, 3);
    auto tensor2 = vt::arange(12, vt::Order::F).reshape(4, 3);
    auto tensor3 = tensor1({1, 3, 1}, {0, 3, 2});
    auto tensor4 = tensor2({1, 3, 1}, {0, 3, 2});
    auto tensor5 = tensor3 + tensor4;
    EXPECT_EQ(vt::asvector(tensor5), (std::vector<float>{2, 3, 10, 11}));

    tensor5 = tensor3 + 1.0f;
    EXPECT_EQ(vt::asvector(tensor5), (std::vector<float>{2, 2, 2, 2}));

    tensor5 = 1.0f + tensor4;
    EXPECT_EQ(vt::asvector(tensor5), (std::vector<float>{2, 3, 10, 11}));

    EXPECT_EQ(vt::asvector(tensor1), (std::vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}));
    EXPECT_EQ(vt::asvector(tensor2), (std::vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}));
}

TEST(SubtractionOperatorForReshapeAndSliceTensorC, BasicAssertions) {
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

TEST(SubtractionOperatorForReshapeAndSliceTensorF, BasicAssertions) {
    auto tensor1 = vt::ones(12, vt::Order::F).reshape(4, 3);
    auto tensor2 = vt::arange(12, vt::Order::F).reshape(4, 3);
    auto tensor3 = tensor1({1, 3, 1}, {0, 3, 2});
    auto tensor4 = tensor2({1, 3, 1}, {0, 3, 2});
    auto tensor5 = tensor4 - tensor3;
    EXPECT_EQ(vt::asvector(tensor5), (std::vector<float>{0, 1, 8, 9}));

    tensor5 = tensor3 - 1.0f;
    EXPECT_EQ(vt::asvector(tensor5), (std::vector<float>{0, 0, 0, 0}));

    tensor5 = 1.0f - tensor3;
    EXPECT_EQ(vt::asvector(tensor5), (std::vector<float>{0, 0, 0, 0}));

    EXPECT_EQ(vt::asvector(tensor1), (std::vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}));
    EXPECT_EQ(vt::asvector(tensor2), (std::vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}));
}

TEST(OperatorMultiplyC, BasicAssertions) {
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

    auto tensor6 = vt::arange(12)({0, 12, 2}).reshape(1, 2, 3);
    auto tensor7 = vt::arange(2).reshape(2, 1, 1);
    auto tensor8 = tensor6 * tensor7;
    EXPECT_EQ(vt::asvector(tensor8), (std::vector<float>{0, 0, 0, 0, 0, 0, 0, 2, 4, 6, 8, 10}));
}

TEST(OperatorMultiplyF, BasicAssertions) {
    auto tensor1 = vt::ones(12, vt::Order::F).reshape(4, 3) + 5.0f;
    auto tensor2 = vt::arange(12, vt::Order::F).reshape(4, 3);
    auto tensor3 = tensor1({1, 3, 1}, {0, 3, 2});
    auto tensor4 = tensor2({1, 3, 1}, {0, 3, 2});
    auto tensor5 = tensor4 * tensor3;
    EXPECT_EQ(vt::asvector(tensor5), (std::vector<float>{6, 12, 54, 60}));

    tensor5 = tensor4 * 6.0f;
    EXPECT_EQ(vt::asvector(tensor5), (std::vector<float>{6, 12, 54, 60}));

    tensor5 = 6.0f * tensor4;
    EXPECT_EQ(vt::asvector(tensor5), (std::vector<float>{6, 12, 54, 60}));

    auto tensor6 = vt::arange(12, vt::Order::F)({0, 12, 2}).reshape(1, 2, 3);
    auto tensor7 = vt::arange(2, vt::Order::F).reshape(2, 1, 1);
    auto tensor8 = tensor6 * tensor7;
    EXPECT_EQ(vt::asvector(tensor8), (std::vector<float>{0, 0, 0, 2, 0, 4, 0, 6, 0, 8, 0, 10}));
}

TEST(OperatorDevideC, BasicAssertions) {
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

    auto tensor6 = vt::arange(12)({0, 12, 2}).reshape(1, 2, 3) + 2.0f;
    auto tensor7 = vt::arange(2).reshape(2, 1, 1);
    auto tensor8 = tensor7 / tensor6;
    EXPECT_EQ(vt::asvector(tensor8), (std::vector<float>{0, 0, 0, 0, 0, 0, 0.5, 0.25, 1.0f/6, 0.125, 0.1, 1.0f/12}));
}

TEST(OperatorDevideF, BasicAssertions) {
    auto tensor1 = vt::ones(12, vt::Order::F).reshape(4, 3) + 5.0f;
    auto tensor2 = vt::arange(12, vt::Order::F).reshape(4, 3);
    auto tensor3 = tensor1({1, 3, 1}, {0, 3, 2});
    auto tensor4 = tensor2({1, 3, 1}, {0, 3, 2});
    auto tensor5 = tensor4 / tensor3;

    auto vec = std::vector<float>{1, 2, 9, 10};
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

    auto tensor6 = vt::arange(12, vt::Order::F)({0, 12, 2}).reshape(1, 2, 3) + 2.0f;
    auto tensor7 = vt::arange(2, vt::Order::F).reshape(2, 1, 1);
    auto tensor8 = tensor7 / tensor6;
    EXPECT_EQ(vt::asvector(tensor8), (std::vector<float>{0, 0.5, 0, 0.25, 0, 1.f/6, 0, 0.125, 0, 0.1, 0, 1.0f/12}));
}

TEST(OperatorGreaterThanC, BasicAssertions) {
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

    auto tensor6 = vt::arange(12)({0, 12, 2}).reshape(1, 2, 3);
    auto tensor7 = vt::arange(2).reshape(2, 1, 1) + 6.0f;
    auto tensor8 = tensor6 > tensor7;
    EXPECT_EQ(vt::asvector(tensor8), (std::vector<bool>{0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1}));
}

TEST(OperatorGreaterThanF, BasicAssertions) {
    auto tensor1 = vt::ones(12, vt::Order::F).reshape(4, 3) + 5.0f;
    auto tensor2 = vt::arange(12, vt::Order::F).reshape(4, 3);
    auto tensor3 = tensor1({1, 3, 1}, {0, 3, 2});
    auto tensor4 = tensor2({1, 3, 1}, {0, 3, 2});
    auto tensor5 = tensor4 > tensor3;
    EXPECT_EQ(vt::asvector(tensor5), (std::vector<bool>{0, 0, 1, 1}));

    tensor5 = tensor4 > 5.0f;
    EXPECT_EQ(vt::asvector(tensor5), (std::vector<bool>{0, 0, 1, 1}));

    tensor5 = 5.0f > tensor4;
    EXPECT_EQ(vt::asvector(tensor5), (std::vector<bool>{1, 1, 0, 0}));

    auto tensor6 = vt::arange(12, vt::Order::F)({0, 12, 2}).reshape(1, 2, 3);
    auto tensor7 = vt::arange(2, vt::Order::F).reshape(2, 1, 1) + 6.0f;
    auto tensor8 = tensor6 > tensor7;
    EXPECT_EQ(vt::asvector(tensor8), (std::vector<bool>{0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1}));
}

TEST(OperatorLessThanC, BasicAssertions) {
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

    auto tensor6 = vt::arange(12)({0, 12, 2}).reshape(1, 2, 3);
    auto tensor7 = vt::arange(2).reshape(2, 1, 1) + 6.0f;
    auto tensor8 = tensor6 < tensor7;
    EXPECT_EQ(vt::asvector(tensor8), (std::vector<bool>{1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0}));
}

TEST(OperatorLessThanF, BasicAssertions) {
    auto tensor1 = vt::ones(12, vt::Order::F).reshape(4, 3) + 5.0f;
    auto tensor2 = vt::arange(12, vt::Order::F).reshape(4, 3);
    auto tensor3 = tensor1({1, 3, 1}, {0, 3, 2});
    auto tensor4 = tensor2({1, 3, 1}, {0, 3, 2});
    auto tensor5 = tensor4 < tensor3;
    EXPECT_EQ(vt::asvector(tensor5), (std::vector<bool>{1, 1, 0, 0}));

    tensor5 = tensor4 < 5.0f;
    EXPECT_EQ(vt::asvector(tensor5), (std::vector<bool>{1, 1, 0, 0}));

    tensor5 = 5.0f < tensor4;
    EXPECT_EQ(vt::asvector(tensor5), (std::vector<bool>{0, 0, 1, 1}));

    auto tensor6 = vt::arange(12, vt::Order::F)({0, 12, 2}).reshape(1, 2, 3);
    auto tensor7 = vt::arange(2, vt::Order::F).reshape(2, 1, 1) + 6.0f;
    auto tensor8 = tensor6 < tensor7;
    EXPECT_EQ(vt::asvector(tensor8), (std::vector<bool>{1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0}));
}

TEST(OperatorGreaterThanOrEqualC, BasicAssertions) {
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

    auto tensor6 = vt::arange(12)({0, 12, 2}).reshape(1, 2, 3);
    auto tensor7 = vt::arange(2).reshape(2, 1, 1) + 6.0f;
    auto tensor8 = tensor6 >= tensor7;
    EXPECT_EQ(vt::asvector(tensor8), (std::vector<bool>{0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1}));
}

TEST(OperatorGreaterThanOrEqualF, BasicAssertions) {
    auto tensor1 = vt::ones(12, vt::Order::F).reshape(4, 3) + 5.0f;
    auto tensor2 = vt::arange(12, vt::Order::F).reshape(4, 3);
    auto tensor3 = tensor1({1, 3, 1}, {0, 3, 2});
    auto tensor4 = tensor2({1, 3, 1}, {0, 3, 2});
    auto tensor5 = tensor4 >= tensor3;
    EXPECT_EQ(vt::asvector(tensor5), (std::vector<bool>{0, 0, 1, 1}));

    tensor5 = tensor4 >= 5.0f;
    EXPECT_EQ(vt::asvector(tensor5), (std::vector<bool>{0, 0, 1, 1}));

    tensor5 = 5.0f >= tensor4;
    EXPECT_EQ(vt::asvector(tensor5), (std::vector<bool>{1, 1, 0, 0}));

    auto tensor6 = vt::arange(12, vt::Order::F)({0, 12, 2}).reshape(1, 2, 3);
    auto tensor7 = vt::arange(2, vt::Order::F).reshape(2, 1, 1) + 6.0f;
    auto tensor8 = tensor6 >= tensor7;
    EXPECT_EQ(vt::asvector(tensor8), (std::vector<bool>{0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1}));
}

TEST(OperatorLessThanOrEqualC, BasicAssertions) {
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

    auto tensor6 = vt::arange(12)({0, 12, 2}).reshape(1, 2, 3);
    auto tensor7 = vt::arange(2).reshape(2, 1, 1) + 6.0f;
    auto tensor8 = tensor6 <= tensor7;
    EXPECT_EQ(vt::asvector(tensor8), (std::vector<bool>{1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0}));
}

TEST(OperatorLessThanOrEqualF, BasicAssertions) {
    auto tensor1 = vt::ones(12, vt::Order::F).reshape(4, 3) + 5.0f;
    auto tensor2 = vt::arange(12, vt::Order::F).reshape(4, 3);
    auto tensor3 = tensor1({1, 3, 1}, {0, 3, 2});
    auto tensor4 = tensor2({1, 3, 1}, {0, 3, 2});
    auto tensor5 = tensor4 <= tensor3;
    EXPECT_EQ(vt::asvector(tensor5), (std::vector<bool>{1, 1, 0, 0}));

    tensor5 = tensor4 <= 5.0f;
    EXPECT_EQ(vt::asvector(tensor5), (std::vector<bool>{1, 1, 0, 0}));

    tensor5 = 5.0f <= tensor4;
    EXPECT_EQ(vt::asvector(tensor5), (std::vector<bool>{0, 0, 1, 1}));

    auto tensor6 = vt::arange(12, vt::Order::F)({0, 12, 2}).reshape(1, 2, 3);
    auto tensor7 = vt::arange(2, vt::Order::F).reshape(2, 1, 1) + 6.0f;
    auto tensor8 = tensor6 <= tensor7;
    EXPECT_EQ(vt::asvector(tensor8), (std::vector<bool>{1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0}));
}

TEST(OperatorEqualC, BasicAssertions) {
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

    auto tensor6 = vt::arange(12)({0, 12, 2}).reshape(1, 2, 3);
    auto tensor7 = vt::arange(2).reshape(2, 1, 1) + 6.0f;
    auto tensor8 = tensor6 == tensor7;
    EXPECT_EQ(vt::asvector(tensor8), (std::vector<bool>{0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0}));
}

TEST(OperatorEqualF, BasicAssertions) {
    auto tensor1 = vt::ones(12, vt::Order::F).reshape(4, 3) + 1.0f;
    auto tensor2 = vt::arange(12, vt::Order::F).reshape(4, 3);
    auto tensor3 = tensor1({1, 3, 1}, {0, 3, 2});
    auto tensor4 = tensor2({1, 3, 1}, {0, 3, 2});
    auto tensor5 = tensor4 == tensor3;
    EXPECT_EQ(vt::asvector(tensor5), (std::vector<bool>{0, 1, 0, 0}));

    tensor5 = tensor4 == 2.0f;
    EXPECT_EQ(vt::asvector(tensor5), (std::vector<bool>{0, 1, 0, 0}));

    tensor5 = 2.0f == tensor4;
    EXPECT_EQ(vt::asvector(tensor5), (std::vector<bool>{0, 1, 0, 0}));

    auto tensor6 = vt::arange(12, vt::Order::F)({0, 12, 2}).reshape(1, 2, 3);
    auto tensor7 = vt::arange(2, vt::Order::F).reshape(2, 1, 1) + 6.0f;
    auto tensor8 = tensor6 == tensor7;
    EXPECT_EQ(vt::asvector(tensor8), (std::vector<bool>{0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0}));
}

TEST(OperatorNotEqualC, BasicAssertions) {
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

    auto tensor6 = vt::arange(12)({0, 12, 2}).reshape(1, 2, 3);
    auto tensor7 = vt::arange(2).reshape(2, 1, 1) + 6.0f;
    auto tensor8 = tensor6 != tensor7;
    EXPECT_EQ(vt::asvector(tensor8), (std::vector<bool>{1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1}));
}

TEST(OperatorNotEqualF, BasicAssertions) {
    auto tensor1 = vt::ones(12, vt::Order::F).reshape(4, 3) + 1.0f;
    auto tensor2 = vt::arange(12, vt::Order::F).reshape(4, 3);
    auto tensor3 = tensor1({1, 3, 1}, {0, 3, 2});
    auto tensor4 = tensor2({1, 3, 1}, {0, 3, 2});
    auto tensor5 = tensor4 != tensor3;
    EXPECT_EQ(vt::asvector(tensor5), (std::vector<bool>{1, 0, 1, 1}));

    tensor5 = tensor4 != 2.0f;
    EXPECT_EQ(vt::asvector(tensor5), (std::vector<bool>{1, 0, 1, 1}));

    tensor5 = 2.0f != tensor4;
    EXPECT_EQ(vt::asvector(tensor5), (std::vector<bool>{1, 0, 1, 1}));

    auto tensor6 = vt::arange(12, vt::Order::F)({0, 12, 2}).reshape(1, 2, 3);
    auto tensor7 = vt::arange(2, vt::Order::F).reshape(2, 1, 1) + 6.0f;
    auto tensor8 = tensor6 != tensor7;
    EXPECT_EQ(vt::asvector(tensor8), (std::vector<bool>{1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1}));
}
