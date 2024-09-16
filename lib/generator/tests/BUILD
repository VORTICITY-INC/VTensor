load("@rules_cuda//cuda:defs.bzl", "cuda_test")

cuda_test(
    name = "test_generator",
    visibility = ["//visibility:public"],
    srcs = [
        "test_arange.cc",
        "test_eye.cc",
        "test_ones.cc",
        "test_zeros.cc",
    ],
    copts = ["-DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE"],
    deps = [
        "//lib:vtensor",
        "@googletest//:gtest_main",
    ],
)
