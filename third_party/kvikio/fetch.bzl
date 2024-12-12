load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def fetch_kvikio():
    http_archive(
        name = "kvikio",
        sha256 = "80a5021691a47d7b9c13958e4638f9c4f819afa68d97f96205b99cd1c2857edf",
        strip_prefix = "kvikio-branch-25.02",
        urls = ["https://github.com/rapidsai/kvikio/archive/refs/heads/branch-25.02.zip"],
        build_file = "//third_party/kvikio:BUILD.bazel",
    )
