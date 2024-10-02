#include <gtest/gtest.h>

#include <lib/vtensor.hpp>

TEST(pinv, BasicAssertions) {
    auto t = vt::arange(12).reshape(3, 2, 2);
    auto tinv = vt::linalg::pinv(t);
    auto re = vt::matmul(t, tinv);
    auto s1 = vt::asvector(re);
    std::vector<float> s2 = {1, 1.6391277e-07, 2.3841858e-07, 9.9999988e-01, 1, -9.5367432e-07, 1.9073486e-06, 1, 1, 0, -3.8146973e-06, 1};
    for (int i = 0; i < s1.size(); i++) {
        EXPECT_NEAR(s1[i], s2[i], 1e-6);
    }
}