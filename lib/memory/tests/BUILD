load("@rules_cuda//cuda:defs.bzl", "cuda_test")

cuda_test(
    name = "test_memory",
    visibility = ["//visibility:public"],
    srcs = [
        "test_ascontiguoustensor.cc",
        "test_asfortrantensor.cc",
        "test_astensor.cc",
        "test_asvector.cc",
        "test_asxarray.cc",
        "test_copy.cc",
        "test_fileio.cc",
    ],
    copts = ["-DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE"],
    deps = [
        "//lib:vtensor",
        "@googletest//:gtest_main",
    ],
)
