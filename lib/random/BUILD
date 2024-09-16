load("@rules_cuda//cuda:defs.bzl", "cuda_library")

cuda_library(
    name = "random",
    visibility = ["//visibility:public"],
    hdrs = [
        "curand.hpp",
        "normal.hpp",
        "rand.hpp",
        ],
    deps = [
        "//lib/core:core",
        "@local_cuda//:curand",
    ],
)