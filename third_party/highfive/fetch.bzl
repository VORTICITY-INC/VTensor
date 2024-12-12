load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def fetch_highfive():
    http_archive(
        name = "highfive",
        sha256 = "1f253cd379038d8541d891ed6faef3922e367349ab66b188c33191c26cf7ae13",
        strip_prefix = "HighFive-master",
        urls = ["https://github.com/BlueBrain/HighFive/archive/refs/heads/master.zip"],
        build_file = "//third_party/highfive:BUILD.bazel",
    )
