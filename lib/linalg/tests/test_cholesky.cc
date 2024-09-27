#include <gtest/gtest.h>

#include <vector>
#include <lib/vtensor.hpp>

TEST(Cholesky, BasicAssertions) {
    std::vector<float> data = {4, 12, -16, 12, 37, -43, -16, -43, 98};
    auto x = vt::astensor(data).reshape(3, 3);
    auto re = vt::linalg::cholesky(x);
    EXPECT_EQ(vt::asvector(re), (std::vector<float>{2, 0, 0, 6, 1, 0, -8, 5, 3}));
}

TEST(BatchedCholesky, BasicAssertions) {
    std::vector<float> data = {4, 12, -16, 12, 37, -43, -16, -43, 98, 4, 12, -16, 12, 37, -43, -16, -43, 98};
    auto x = vt::astensor(data).reshape(2, 3, 3);
    auto re = vt::linalg::cholesky(x);
    EXPECT_EQ(vt::asvector(re), (std::vector<float>{2, 0, 0, 6, 1, 0, -8, 5, 3, 2, 0, 0, 6, 1, 0, -8, 5, 3}));
}
