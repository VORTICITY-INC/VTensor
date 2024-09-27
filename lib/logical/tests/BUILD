load("@rules_cuda//cuda:defs.bzl", "cuda_test")

cuda_test(
    name = "test_logical",
    visibility = ["//visibility:public"],
    srcs = [
        "test_all.cc",
        "test_any.cc",
        "test_where.cc",
    ],
    copts = ["-DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE"],
    deps = [
        "//lib:vtensor",
        "@googletest//:gtest_main",
    ],
)
