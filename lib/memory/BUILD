load("@rules_cuda//cuda:defs.bzl", "cuda_library")

cuda_library(
    name = "memory",
    visibility = ["//visibility:public"],
    hdrs = [
        "ascontiguoustensor.hpp",
        "asfortrantensor.hpp",
        "astensor.hpp",
        "asvector.hpp",
        "asxarray.hpp",
        "copy.hpp",
        "fileio.hpp",
    ],
    deps = [
        "//lib/core:core",
        "//lib/linalg:linalg",
        "@xtensor",
    ],
)