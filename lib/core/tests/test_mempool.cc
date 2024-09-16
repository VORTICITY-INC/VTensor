#include <gtest/gtest.h>

#include <lib/vtensor.hpp>

TEST(GlobalMempoolTest, BasicAssertions) {
    vt::GlobalMempool& instance1 = vt::GlobalMempool::get_instance();
    vt::GlobalMempool& instance2 = vt::GlobalMempool::get_instance();
    EXPECT_EQ(&instance1, &instance2);
}