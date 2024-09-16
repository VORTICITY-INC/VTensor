#pragma once

#include <cuda_runtime.h>

namespace vt {
namespace time {

/**
 * @brief Timer class to measure the time of a CUDA kernel.
 */
class GPUTimer {
   public:
    /**
     * @brief Construct a new GPUTimer object.
     */
    GPUTimer() {
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
    }

    /**
     * @brief Destroy the GPUTimer object.
     */
    ~GPUTimer() {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }

    /**
     * @brief Start the timer.
     */
    void start() { cudaEventRecord(start_event, 0); }

    /**
     * @brief Stop the timer.
     *
     * @return float: The time in milliseconds.
     */
    float stop() {
        cudaEventRecord(stop_event, 0);
        cudaEventSynchronize(stop_event);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start_event, stop_event);
        return milliseconds;
    }

   private:
    cudaEvent_t start_event;
    cudaEvent_t stop_event;
};

}  // namespace time
}  // namespace vt
