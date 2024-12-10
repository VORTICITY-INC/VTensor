load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def fetch_xtl():
    http_archive(
        name = "xtl",
        sha256 = "ce4ec05202e8211ec84ae16c19ed5a3bba1344db65e3495f0d83569babb1c21f",
        strip_prefix = "xtl-master",
        urls = ["https://github.com/xtensor-stack/xtl/archive/refs/heads/master.zip"],
        build_file = "//third_party/xtl:BUILD.bazel",
    )
