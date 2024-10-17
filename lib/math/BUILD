load("@rules_cuda//cuda:defs.bzl", "cuda_library")

cuda_library(
    name = "math",
    visibility = ["//visibility:public"],
    hdrs = [
        "exp.hpp",
        "expand_dims.hpp",
        "max.hpp",
        "maximum.hpp",
        "mean.hpp",
        "min.hpp",
        "minimum.hpp",
        "power.hpp",
        "reduce.hpp",
        "sort.hpp",
        "sqrt.hpp",
        "sum.hpp",
        "transpose.hpp",
        "vander.hpp",
    ],
    deps = [
        "//lib/core:core",
        "//lib/memory:memory",
    ],
)

