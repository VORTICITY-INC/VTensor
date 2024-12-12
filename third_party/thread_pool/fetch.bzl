load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def fetch_thread_pool():
    http_archive(
        name = "thread_pool",
        sha256 = "bd396440eeab4b55c3e8798b5baded455025385e241dc4a8c0f7a17705c98ee2",
        strip_prefix = "thread-pool-master",
        urls = ["https://github.com/bshoshany/thread-pool/archive/refs/heads/master.zip"],
        build_file = "//third_party/thread_pool:BUILD.bazel",
    )
