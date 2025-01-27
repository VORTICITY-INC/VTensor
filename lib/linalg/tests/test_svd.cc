#include <gtest/gtest.h>

#include <lib/vtensor.hpp>

TEST(SVDFullMatrix1, BasicAssertions) {
    auto tensor = vt::arange(12).reshape(4, 3);
    auto [vt, s, u] = vt::linalg::svd(tensor);

    auto s1 = vt::asvector(s);
    std::vector<float> s2 = {2.2446751e+01, 1.4640590e+00, 5.8867556e-07};
    for (int i = 0; i < s1.size(); i++) {
        EXPECT_NEAR(s1[i], s2[i], 1e-6);
    }

    auto vt1 = vt::asvector(vt);
    std::vector<float> vt2 = {
        -0.08351804,  0.83248144,  0.5426417,  -0.0744282,
        -0.31365088,  0.44902438, -0.6680477,   0.503699,
        -0.54378384,  0.06556801, -0.2918319,  -0.7841137,
        -0.7739167,  -0.31788802,  0.41723716,  0.35484284
    };
    for (int i = 0; i < vt1.size(); i++) {
        EXPECT_NEAR(vt1[i], vt2[i], 1e-6);
    }

    auto u1 = vt::asvector(u);
    std::vector<float> u2 = {
        -0.49757335, -0.5739708,  -0.65036786,
        -0.7653458,  -0.06237787,  0.6405894,
        -0.40824816,  0.81649655, -0.4082485
    };
    for (int i = 0; i < u1.size(); i++) {
        EXPECT_NEAR(u1[i], u2[i], 1e-6);
    }
}

TEST(SVDFullMatrix2, BasicAssertions) {
    auto tensor = vt::arange(12).reshape(3, 4);
    auto [vt, s, u] = vt::linalg::svd(tensor);

    auto s1 = vt::asvector(s);
    std::vector<float> s2 = {2.2409298e+01, 1.9553399e+00, 3.9066444e-07};
    for (int i = 0; i < s1.size(); i++) {
        EXPECT_NEAR(s1[i], s2[i], 1e-6);
    }

    auto vt1 = vt::asvector(vt);
    std::vector<float> vt2 = {
        -0.1473065,  -0.9009075,   0.40824804,
        -0.50027525, -0.28819758, -0.8164966,
        -0.853244,    0.32451168,  0.40824842
    };
    for (int i = 0; i < vt1.size(); i++) {
        EXPECT_NEAR(vt1[i], vt2[i], 1e-6);
    }

    auto u1 = vt::asvector(u);
    std::vector<float> u2 = {
        -0.39390135, -0.4608747,  -0.52784806, -0.59482133,
         0.7381339,   0.2959636,  -0.14620665, -0.58837706,
         0.42559415, -0.82443875,  0.37209475,  0.02674968,
         0.34477443, -0.1424799,  -0.74936336,  0.54706883
    };
    for (int i = 0; i < u1.size(); i++) {
        EXPECT_NEAR(u1[i], u2[i], 1e-6);
    }
}

TEST(SVDNotFullMatrix1, BasicAssertions) {
    auto tensor = vt::arange(12).reshape(4, 3);
    auto [vt, s, u] = vt::linalg::svd(tensor, false);

    auto s1 = vt::asvector(s);
    std::vector<float> s2 = {2.2446751e+01, 1.4640590e+00, 5.8867556e-07};
    for (int i = 0; i < s1.size(); i++) {
        EXPECT_NEAR(s1[i], s2[i], 1e-6);
    }

    auto vt1 = vt::asvector(vt);
    std::vector<float> vt2 = {
        -0.08351804,  0.83248144,  0.5426417,
        -0.31365088,  0.44902438, -0.6680477,
        -0.54378384,  0.06556801, -0.2918319,
        -0.7739167,  -0.31788802,  0.41723716,
    };

    for (int i = 0; i < vt1.size(); i++) {
        EXPECT_NEAR(vt1[i], vt2[i], 1e-6);
    }

    auto u1 = vt::asvector(u);
    std::vector<float> u2 = {
        -0.49757335, -0.5739708,  -0.65036786,
        -0.7653458,  -0.06237787,  0.6405894,
        -0.40824816,  0.81649655, -0.4082485
    };
    for (int i = 0; i < u1.size(); i++) {
        EXPECT_NEAR(u1[i], u2[i], 1e-6);
    }
}

TEST(SVDNotFullMatrix2, BasicAssertions) {
    auto tensor = vt::arange(12).reshape(3, 4);
    auto [vt, s, u] = vt::linalg::svd(tensor, false);

    auto s1 = vt::asvector(s);
    std::vector<float> s2 = {2.2409298e+01, 1.9553399e+00, 3.9066444e-07};
    for (int i = 0; i < s1.size(); i++) {
        EXPECT_NEAR(s1[i], s2[i], 1e-6);
    }

    auto vt1 = vt::asvector(vt);
    std::vector<float> vt2 = {
        -0.1473065,  -0.9009075,   0.40824804,
        -0.50027525, -0.28819758, -0.8164966,
        -0.853244,    0.32451168,  0.40824842
    };
    for (int i = 0; i < vt1.size(); i++) {
        EXPECT_NEAR(vt1[i], vt2[i], 1e-6);
    }

    auto u1 = vt::asvector(u);
    std::vector<float> u2 = {
        -0.39390135, -0.4608747,  -0.52784806, -0.59482133,
         0.7381339,   0.2959636,  -0.14620665, -0.58837706,
         0.42559415, -0.82443875,  0.37209475,  0.02674968,
    };
    for (int i = 0; i < u1.size(); i++) {
        EXPECT_NEAR(u1[i], u2[i], 1e-6);
    }
}

TEST(SVDNotComputUV1, BasicAssertions) {
    auto tensor = vt::arange(12).reshape(4, 3);
    auto [vt, s, u] = vt::linalg::svd(tensor, false, false);
    auto s1 = vt::asvector(s);
    std::vector<float> s2 = {2.2446751e+01, 1.4640590e+00, 5.8867556e-07};
    for (int i = 0; i < s1.size(); i++) {
        EXPECT_NEAR(s1[i], s2[i], 1e-6);
    }
}

TEST(SVDNotComputUV2, BasicAssertions) {
    auto tensor = vt::arange(12).reshape(3, 4);
    auto [vt, s, u] = vt::linalg::svd(tensor, false, false);
    auto s1 = vt::asvector(s);
    std::vector<float> s2 = {2.2409296e+01, 1.9553399e+00, 3.9066444e-07};
    for (int i = 0; i < s1.size(); i++) {
        EXPECT_NEAR(s1[i], s2[i], 1e-6);
    }
}

TEST(BatchedSVDFullMatrix, BasicAssertions) {
    auto tensor = vt::arange(12).reshape(2, 3, 2);
    auto [vt, s, u] = vt::linalg::svd(tensor);
    auto s1 = vt::asvector(s);
    std::vector<float> s2 = {7.386483, 0.66323584, 21.235508, 0.23069805};
    for (int i = 0; i < s1.size(); i++) {
        EXPECT_NEAR(s1[i], s2[i], 1e-6);
    }

    auto vt1 = vt::asvector(vt);
    std::vector<float> vt2 = {
        -0.10818539, -0.9064377,  0.40824834,
        -0.48733622, -0.30957523, -0.81649655,
        -0.86648697,  0.28728768,  0.40824825,
        -0.43406904, -0.8030684,  0.40824732,
        -0.5670487,  -0.10857622, -0.8164968,
        -0.7000285,   0.58591187,  0.4082491
    };
    for (int i = 0; i < vt1.size(); i++) {
        EXPECT_NEAR(vt1[i], vt2[i], 1e-6);
    }

    auto u1 = vt::asvector(u);
    std::vector<float> u2 = {
        -0.6011819,  -0.79911214,  0.79911214, -0.6011819,
        -0.66591716, -0.74602574,  0.74602574, -0.66591716
    };
    for (int i = 0; i < u1.size(); i++) {
        EXPECT_NEAR(u1[i], u2[i], 1e-6);
    }
}

TEST(BatchedSVDNotFullMatrix, BasicAssertions) {
    auto tensor = vt::arange(12).reshape(2, 3, 2);
    auto [vt, s, u] = vt::linalg::svd(tensor, false);
    auto s1 = vt::asvector(s);
    std::vector<float> s2 = {7.386483, 0.66323584, 21.235508, 0.23069805};
    for (int i = 0; i < s1.size(); i++) {
        EXPECT_NEAR(s1[i], s2[i], 1e-6);
    }

    auto vt1 = vt::asvector(vt);
    std::vector<float> vt2 = {
        -0.10818539, -0.9064377,  -0.48733622, -0.30957523, -0.86648697,  0.28728768,
        -0.43406904, -0.8030684,  -0.5670487,  -0.10857622, -0.7000285,   0.58591187
    };
    for (int i = 0; i < vt1.size(); i++) {
        EXPECT_NEAR(vt1[i], vt2[i], 1e-6);
    }

    auto u1 = vt::asvector(u);
    std::vector<float> u2 = {
        -0.6011819,  -0.79911214,  0.79911214, -0.6011819,
        -0.66591716, -0.74602574,  0.74602574, -0.66591716
    };
    for (int i = 0; i < u1.size(); i++) {
        EXPECT_NEAR(u1[i], u2[i], 1e-6);
    }
}
