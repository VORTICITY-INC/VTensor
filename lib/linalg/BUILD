load("@rules_cuda//cuda:defs.bzl", "cuda_library")

cuda_library(
    name = "linalg",
    visibility = ["//visibility:public"],
    hdrs = [
        "cholesky.hpp",
        "cublas.hpp",
        "cusolver.hpp",
        "inv.hpp",
        "matmul.hpp",
        "product.hpp",
        "svd.hpp",
    ],
    deps = [
        "//lib/core:core",
        "@local_cuda//:cublas",
        "@local_cuda//:cusolver",
    ],
)

