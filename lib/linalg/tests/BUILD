load("@rules_cuda//cuda:defs.bzl", "cuda_test")

cuda_test(
    name = "test_linalg",
    visibility = ["//visibility:public"],
    srcs = [
        "test_cublas.cc",
        "test_cusolver.cc",
        "test_inv.cc",
        "test_matmul.cc",
        "test_product.cc",
        "test_svd.cc",
    ],
    copts = ["-DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE"],
    deps = [
        "//lib:vtensor",
        "@googletest//:gtest_main",
    ],
)
