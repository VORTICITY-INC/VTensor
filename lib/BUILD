load("@rules_cuda//cuda:defs.bzl", "cuda_library")

cuda_library(
    name = "vtensor",
    visibility = ["//visibility:public"],
    hdrs = ["vtensor.hpp"],
    deps = [
        "//lib/core:core",
        "//lib/generator:generator",
        "//lib/linalg:linalg",
        "//lib/logical:logical",
        "//lib/math:math",
        "//lib/memory:memory",
        "//lib/random:random",
        "//lib/time:time",
    ],
)
