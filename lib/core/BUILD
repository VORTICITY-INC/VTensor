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
        "operator.hpp",
        "print.hpp",
        "slice.hpp",
        "tensor.hpp",
    ],
    deps = [
        "@local_cuda//:thrust",
        "@rmm",
    ],
)

cuda_library(
    name = "mempool",
    visibility = ["//visibility:public"],
    hdrs = ["mempool.hpp"],
    deps = [
        ":rmm_libs",
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

exports_files(
    ["rmm_utils.hpp", "rmm_utils.cc"],
    visibility = ["//visibility:public"],
)