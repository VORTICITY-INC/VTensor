#include <gtest/gtest.h>

#include "lib/core/mempool.hpp"

TEST(GlobalMempoolTest, BasicAssertions) {
    vt::GlobalMempool& instance1 = vt::GlobalMempool::get_instance(1);
    vt::GlobalMempool& instance2 = vt::GlobalMempool::get_instance(1);
    EXPECT_EQ(&instance1, &instance2);
}