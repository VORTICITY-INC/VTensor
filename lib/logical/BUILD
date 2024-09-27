load("@rules_cuda//cuda:defs.bzl", "cuda_library")

cuda_library(
    name = "logical",
    visibility = ["//visibility:public"],
    hdrs = [
        "all.hpp",
        "any.hpp",
        "where.hpp",
    ],
    deps = [
        "//lib/core:core",
        "//lib/math:math",
    ],
)

