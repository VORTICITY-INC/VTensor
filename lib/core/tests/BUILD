load("@rules_cuda//cuda:defs.bzl", "cuda_test")

cuda_test(
    name = "test_core",
    visibility = ["//visibility:public"],
    srcs = [
        "test_broadcast.cc",
        "test_cutensor.cc",
        "test_iterator.cc",
        "test_mempool.cc",
        "test_operator.cc",
        "test_slice.cc",
        "test_tensor.cc"
    ],
    copts = ["-DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE"],
    deps = [
        "//lib:vtensor",
        "//lib/core:mempool",
        "@googletest//:gtest_main",
    ],
)
