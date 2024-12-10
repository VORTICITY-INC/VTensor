load("//third_party/nvtx:fetch.bzl", "fetch_nvtx")
load("//third_party/rmm:fetch.bzl", "fetch_rmm")
load("//third_party/xtl:fetch.bzl", "fetch_xtl")
load("//third_party/xtensor:fetch.bzl", "fetch_xtensor")

def fetch_deps():
    fetch_nvtx()
    fetch_rmm()
    fetch_xtl()
    fetch_xtensor()
