#include <gtest/gtest.h>

#include <lib/vtensor.hpp>

TEST(Exp, BasicAssertions) {
    auto tensor = vt::arange(6);
    auto re = vt::exp(tensor);
    auto s1 = vt::asvector(re);
    auto s2 = std::vector<float>{1, 2.71828183, 7.3890561, 20.08553692, 54.59815003, 148.4131591};
    for (auto i = 0; i < re.size(); ++i) {
        EXPECT_NEAR(s1[i], s2[i], 1e-5);
    }
}
