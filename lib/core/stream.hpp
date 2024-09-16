#pragma once

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>

namespace vt {
// This is just a wrapper for rmm::cuda_stream and rmm::cuda_stream_view
using cuda_stream = rmm::cuda_stream;
using cuda_stream_view = rmm::cuda_stream_view;
static constexpr cuda_stream_view cuda_stream_default{};

}  // namespace vt
