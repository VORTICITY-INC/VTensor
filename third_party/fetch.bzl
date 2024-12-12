load("//third_party/nvtx:fetch.bzl", "fetch_nvtx")
load("//third_party/rmm:fetch.bzl", "fetch_rmm")
load("//third_party/xtl:fetch.bzl", "fetch_xtl")
load("//third_party/xtensor:fetch.bzl", "fetch_xtensor")
load("//third_party/thread_pool:fetch.bzl", "fetch_thread_pool")
load("//third_party/cnpy:fetch.bzl", "fetch_cnpy")
load("//third_party/highfive:fetch.bzl", "fetch_highfive")
load("//third_party/kvikio:fetch.bzl", "fetch_kvikio")

def fetch_deps():
    fetch_nvtx()
    fetch_rmm()
    fetch_xtl()
    fetch_xtensor()
    fetch_thread_pool()
    fetch_cnpy()
    fetch_highfive()
    fetch_kvikio()
