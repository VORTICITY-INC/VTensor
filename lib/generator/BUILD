load("@rules_cuda//cuda:defs.bzl", "cuda_library")

cuda_library(
    name = "generator",
    visibility = ["//visibility:public"],
    hdrs = [
        "arange.hpp",
        "diag.hpp",
        "eye.hpp",
        "ones.hpp",
        "tri.hpp",
        "zeros.hpp", 
    ],
    deps = [
        "//lib/core:core",
    ],
)