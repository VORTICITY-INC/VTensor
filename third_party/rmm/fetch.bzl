load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def fetch_rmm():
    http_archive(
        name = "rmm",
        sha256 = "00e74a0445042c489b8174b6eaa759d41655561c317e9a04ab4a73cddde619c0",
        strip_prefix = "rmm-24.08.00",
        urls = ["https://github.com/rapidsai/rmm/archive/refs/tags/v24.08.00.zip"],
        build_file = "//third_party/rmm:BUILD.bazel",
    )
