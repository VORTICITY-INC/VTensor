load("@rules_cuda//cuda:defs.bzl", "cuda_library")

cuda_library(
    name = "core",
    visibility = ["//visibility:public"],
    hdrs = [
        "assertions.hpp",
        "astype.hpp",
        "broadcast.hpp",
        "cutensor.hpp",
        "iterator.hpp",
        "mempool.hpp",
        "operator.hpp",
        "print.hpp",
        "slice.hpp",
        "tensor.hpp",
    ],
    deps = [
        "//lib/core:rmm_libs",
        "@local_cuda//:thrust",
        "@rmm",
    ],
)

cc_library(
    name = "rmm_libs",
    visibility = ["//visibility:public"],
    srcs = ["rmm_utils.so"],
    hdrs = ["rmm_utils.hpp"],
)

cc_binary(
    name = "rmm_utils.so",
    visibility = ["//visibility:public"],
    srcs = ["rmm_utils.cc"],
    copts = ["-DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE"],
    deps = [":rmm_utils"],
    linkshared = True,
)

cuda_library(
    name = "rmm_utils",
    visibility = ["//visibility:public"],
    hdrs = ["rmm_utils.hpp"],
    deps = ["@rmm"],
)
