load("@rules_cuda//cuda:defs.bzl", "cuda_test")

cuda_test(
    name = "test_math",
    visibility = ["//visibility:public"],
    srcs = [
        "test_exp.cc",
        "test_expand_dims.cc",
        "test_max.cc",
        "test_maximum.cc",
        "test_mean.cc",
        "test_min.cc",
        "test_minimum.cc",
        "test_power.cc",
        "test_sort.cc",
        "test_sqrt.cc",
        "test_sum.cc",
        "test_transpose.cc",
        "test_vander.cc",
    ],
    copts = ["-DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE"],
    deps = [
        "//lib:vtensor",
        "@googletest//:gtest_main",
    ],
)
