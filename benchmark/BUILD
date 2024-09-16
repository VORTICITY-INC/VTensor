load("@rules_cuda//cuda:defs.bzl", "cuda_binary")

cuda_binary(
    name = "benchmark",
    visibility = ["//visibility:public"],
    srcs = ["benchmark.cc"],
    copts = ["-DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE"],
    deps = [
        "//lib:vtensor",
        "@google_benchmark//:benchmark_main"
    ],
)
