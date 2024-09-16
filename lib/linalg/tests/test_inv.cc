#include <gtest/gtest.h>

#include <vector>
#include <lib/vtensor.hpp>

TEST(Inv2D, BasicAssertions) {
    auto tensor = vt::arange(4).reshape(2, 2);

    auto handle = vt::cuda::cusolver.get_handle();
    auto re = vt::linalg::inv(tensor, handle);
    EXPECT_EQ(vt::asvector(tensor), (std::vector<float>{0, 1, 2, 3}));
    EXPECT_EQ(vt::asvector(re), (std::vector<float>{-1.5, 0.5, 1, 0}));
}

TEST(Inv3D, BasicAssertions) {
    auto vec = std::vector<float>{0, 1, 2, 3, 0, 1, 2, 3};
    auto tensor = vt::astensor(vec).reshape(2, 2, 2);
    auto re = vt::linalg::inv(tensor);
    EXPECT_EQ(vt::asvector(tensor), (std::vector<float>{0, 1, 2, 3, 0, 1, 2, 3}));
    EXPECT_EQ(vt::asvector(re), (std::vector<float>{-1.5, 0.5, 1, 0, -1.5, 0.5, 1, 0}));
}
