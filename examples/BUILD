load("@rules_cuda//cuda:defs.bzl", "cuda_binary")

cuda_binary(
    name = "cutensor",
    visibility = ["//visibility:public"],
    srcs = ["cutensor.cc"],
    copts = ["-DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE"],
    deps = [
        "//lib:vtensor",
        "@xtensor",
        "@thread_pool",
    ],
)

cuda_binary(
    name = "monte_carlo",
    visibility = ["//visibility:public"],
    srcs = ["monte_carlo.cc"],
    copts = ["-DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE"],
    deps = [
        "//lib:vtensor",
    ],
)
