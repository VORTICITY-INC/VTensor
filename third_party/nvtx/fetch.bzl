load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def fetch_nvtx():
    http_archive(
        name = "nvtx",
        sha256 = "dbf307ce0e5b0f6ceacf37d65ca7a70e9a545d4162f824e8c0ac7b86f5640aa4",
        strip_prefix = "NVTX-3.1.0",
        urls = ["https://github.com/NVIDIA/NVTX/archive/refs/tags/v3.1.0.zip"],
        build_file = "//third_party/nvtx:BUILD.bazel",
    )
