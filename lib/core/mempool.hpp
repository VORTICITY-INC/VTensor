#pragma once

#ifndef POOL_SIZE
#define POOL_SIZE 50
#endif

#ifndef DISABLE_GLOBAL_MEMPOOL
#define DISABLE_GLOBAL_MEMPOOL false
#endif

#include "lib/core/rmm_utils.hpp"

namespace vt {

/**
 * @brief Global memory pool initializer for VTensor.
 */
class GlobalMempoolInitializer {
   public:
    GlobalMempoolInitializer() {
        if (!DISABLE_GLOBAL_MEMPOOL) {
            GlobalMempool::get_instance(POOL_SIZE);
        }
    }
};

static GlobalMempoolInitializer mempool_initializer;

}  // namespace vt
