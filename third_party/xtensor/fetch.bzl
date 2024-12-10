load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def fetch_xtensor():
    http_archive(
        name = "xtensor",
        sha256 = "627705b29d2d9395d36cc233f88a2fcb32b544460a53b5deddc886a6346d80dd",
        strip_prefix = "xtensor-master",
        urls = ["https://github.com/xtensor-stack/xtensor/archive/refs/heads/master.zip"],
        build_file = "//third_party/xtensor:BUILD.bazel",
    )
