#include <gtest/gtest.h>

#include <lib/vtensor.hpp>

TEST(ArangeGeneratorC, BasicAssertions) {
    auto tensor = vt::arange(12);
    EXPECT_EQ(vt::asvector(tensor), (std::vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}));

    auto tensor1 = vt::arange<int>(1, 12, 2);
    EXPECT_EQ(vt::asvector(tensor1), (std::vector<int>{1, 3, 5, 7, 9, 11}));

    float arr[3] = {0.0f, 1.0f, 2.0f};
    float* ptr = &arr[0];
    auto tensor2 = vt::arange(ptr, ptr + 3, 2);
    EXPECT_EQ(vt::asvector(tensor2), (std::vector<float*>{ptr, ptr + 2}));

    auto tensor3 = vt::arange(1.2f, 11.2f, 2.0f);
    EXPECT_EQ(vt::asvector(tensor3), (std::vector<float>{1.2, 3.2, 5.2, 7.2, 9.2}));

    auto tensor4 = vt::arange(0.0f, 1.0f, 0.3f);
    auto vec = vt::asvector(tensor4);
    EXPECT_EQ(vec.size(), 4);
    for (auto i = 0; i < 4; ++i) EXPECT_NEAR(vec[i], i * 0.3, 1e-6);

    auto tensor5 = vt::arange(9.2f, 1.0f, -2.0f);
    vec = vt::asvector(tensor5);
    EXPECT_EQ(vec.size(), 5);
    for (auto i = 0; i < 5; ++i) EXPECT_NEAR(vec[i], 9.2 - i * 2.0, 1e-6);

    auto tensor6 = vt::arange(0.9f, 0.1f, -0.3f);
    vec = vt::asvector(tensor6);
    EXPECT_EQ(vec.size(), 3);
    for (auto i = 0; i < 3; ++i) EXPECT_NEAR(vec[i], 0.9 - i * 0.3, 1e-6);
}

TEST(ArangeGeneratorF, BasicAssertions) {
    auto tensor = vt::arange(12, vt::Order::F);
    EXPECT_EQ(vt::asvector(tensor), (std::vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}));

    auto tensor1 = vt::arange<int>(1, 12, 2, vt::Order::F);
    EXPECT_EQ(vt::asvector(tensor1), (std::vector<int>{1, 3, 5, 7, 9, 11}));

    float arr[3] = {0.0f, 1.0f, 2.0f};
    float* ptr = &arr[0];
    auto tensor2 = vt::arange(ptr, ptr + 3, 2, vt::Order::F);
    EXPECT_EQ(vt::asvector(tensor2), (std::vector<float*>{ptr, ptr + 2}));

    auto tensor3 = vt::arange(1.2f, 11.2f, 2.0f, vt::Order::F);
    EXPECT_EQ(vt::asvector(tensor3), (std::vector<float>{1.2, 3.2, 5.2, 7.2, 9.2}));

    auto tensor4 = vt::arange(0.0f, 1.0f, 0.3f, vt::Order::F);
    auto vec = vt::asvector(tensor4);
    EXPECT_EQ(vec.size(), 4);
    for (auto i = 0; i < 4; ++i) EXPECT_NEAR(vec[i], i * 0.3, 1e-6);

    auto tensor5 = vt::arange(9.2f, 1.0f, -2.0f, vt::Order::F);
    vec = vt::asvector(tensor5);
    EXPECT_EQ(vec.size(), 5);
    for (auto i = 0; i < 5; ++i) EXPECT_NEAR(vec[i], 9.2 - i * 2.0, 1e-6);

    auto tensor6 = vt::arange(0.9f, 0.1f, -0.3f, vt::Order::F);
    vec = vt::asvector(tensor6);
    EXPECT_EQ(vec.size(), 3);
    for (auto i = 0; i < 3; ++i) EXPECT_NEAR(vec[i], 0.9 - i * 0.3, 1e-6);
}