load("@rules_cuda//cuda:defs.bzl", "cuda_test")

cuda_test(
    name = "test_random",
    visibility = ["//visibility:public"],
    srcs = [
        "test_curand.cc",
        "test_normal.cc",
        "test_rand.cc",
    ],
    copts = ["-DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE"],
    deps = [
        "//lib:vtensor",
        "@googletest//:gtest_main",
    ],
)
