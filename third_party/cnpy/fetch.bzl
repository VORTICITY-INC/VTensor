load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def fetch_cnpy():
    http_archive(
        name = "cnpy",
        sha256 = "dd58304295ce30e2299fafe8a72cb1acac508139df740c8976bcaab5c17591dc",
        strip_prefix = "cnpy-master",
        urls = ["https://github.com/rogersce/cnpy/archive/refs/heads/master.zip"],
        build_file = "//third_party/cnpy:BUILD.bazel",
    )
