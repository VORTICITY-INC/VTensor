load("//third_party/cnpy:fetch.bzl", "fetch_cnpy")
load("//third_party/nvtx:fetch.bzl", "fetch_nvtx")
load("//third_party/rmm:fetch.bzl", "fetch_rmm")

def fetch_deps():
    fetch_cnpy()
    fetch_nvtx()
    fetch_rmm()
