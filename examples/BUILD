load("@rules_cuda//cuda:defs.bzl", "cuda_binary")

cuda_binary(
    name = "cutensor",
    visibility = ["//visibility:public"],
    srcs = ["cutensor.cc"],
    copts = ["-DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE"],
    deps = [
        "//lib:vtensor",
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

cuda_binary(
    name = "checkpoint",
    visibility = ["//visibility:public"],
    srcs = ["checkpoint.cc"],
    copts = [
        "-DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE",
        "-I/usr/include/hdf5/serial",
    ],
    deps = [
        "//lib:vtensor",
        "@cnpy",
        "@xtensor",
        "@thread_pool",
        "@highfive",
        "@kvikio", 
    ],
    linkopts = ["-L/usr/lib/x86_64-linux-gnu/hdf5/serial"],
)

cuda_binary(
    name = "io",
    visibility = ["//visibility:public"],
    srcs = ["io_test.cc"],
    copts = [
        "-DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE",
        "-I/usr/include/hdf5/serial",
    ],
    deps = [
        "//lib:vtensor",
        "@cnpy",
        "@xtensor",
        "@thread_pool",
        "@highfive",
        "@kvikio", 
    ],
    linkopts = ["-L/usr/lib/x86_64-linux-gnu/hdf5/serial"],
)
