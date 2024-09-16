load("@rules_cuda//cuda:defs.bzl", "cuda_library")

cuda_library(
    name = "core",
    visibility = ["//visibility:public"],
    hdrs = [
        "cutensor.hpp",
        "iterator.hpp",
        "mempool.hpp",
        "operator.hpp",
        "print.hpp",
        "slice.hpp",
        "stream.hpp",
        "tensor.hpp",
    ],
    deps = [
        "@rmm",
        "@local_cuda//:thrust",
    ],
)

