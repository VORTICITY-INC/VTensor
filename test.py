import cupy as cp
import numpy as np
from concurrent.futures import Future, ThreadPoolExecutor
from cupyx import empty_pinned
import time

num_files_per_gpu = 10
num_gpus = 1
file_size_gb = 1
dtype = cp.uint8
shape = int(file_size_gb * 1e9)
total_files = num_files_per_gpu * num_gpus
total_size = file_size_gb * total_files


def create_gpu_data():
    data = []
    for d in range(num_gpus):
        for f in range(num_files_per_gpu):
            with cp.cuda.Device(d):
                data.append(cp.random.randint(0, 256, size=shape, dtype=dtype))
    return data

def from_device_to_ram():
    data = create_gpu_data()
    streams = [cp.cuda.Stream(non_blocking=True) for i in range(total_files)]

    def job(data, stream):
        with stream:
            a = cp.asnumpy(data)
    pool = ThreadPoolExecutor(max_workers=total_files)
    for i in range(len(data)):
        pool.submit(job, data[i], streams[i])

    tic = time.time()
    pool.shutdown(wait=True)
    toc = time.time()
    duration = toc - tic
    badwidth = total_size / duration
    print(f"From device to RAM {badwidth}GB/s")

def from_device_to_pinned_memory():
    data = create_gpu_data()
    tic = time.time()
    pinned_mems = []
    for i in range(total_files):
        pinned_mems.append(empty_pinned(shape, dtype=dtype))
    toc = time.time()
    duration = toc - tic
    print(f"Create pinned memory {duration}s")
        
    streams = [cp.cuda.Stream(non_blocking=True) for i in range(total_files)]
    def job(data, pinned_mem, stream):
        data.get(out=pinned_mem, stream=stream)
        stream.synchronize()

    pool = ThreadPoolExecutor(max_workers=total_files)
    for i in range(len(data)):
        pool.submit(job, data[i], pinned_mems[i], streams[i])

    tic = time.time()
    pool.shutdown(wait=True)
    toc = time.time()
    duration = toc - tic
    badwidth = total_size / duration
    print(f"From device to pinned memory {badwidth}GB/s")

    np.testing.assert_equal(cp.asnumpy(data[0]), pinned_mems[0])

def from_device_to_disk():
    data = create_gpu_data()
    mmap_files = [np.memmap(f"/mammoth/tmp/{i}.npy", dtype=dtype, mode='w+', shape=(shape)) for i in range(total_files)]
    def job(data, mmap):
        data.get(out=mmap)

    pool = ThreadPoolExecutor(max_workers=total_files)
    for i in range(len(data)):
        pool.submit(job, data[i], mmap_files[i])

    tic = time.time()
    pool.shutdown(wait=True)
    toc = time.time()
    duration = toc - tic
    badwidth = total_size / duration
    print(f"From device to disk {badwidth}GB/s")

    np.testing.assert_equal(cp.asnumpy(data[0]), mmap_files[0])


def create_host_data():
    data = []
    for f in range(num_files_per_gpu):
        data.append(np.random.randint(0, 256, size=shape, dtype=dtype))
    return data


def from_host_to_disk():
    data = create_host_data()
    # mmap_files = [np.memmap(f"/mammoth/tmp/{i}.npy", dtype=dtype, mode='w+', shape=(shape)) for i in range(total_files)]
    def job(data, i):
        np.save(f"/mammoth/checkpoints/{i}.npy", data)
    
    tic = time.time()
    pool = ThreadPoolExecutor(max_workers=total_files)
    for i in range(len(data)):
        pool.submit(job, data[i], i)

    
    pool.shutdown(wait=True)
    toc = time.time()
    duration = toc - tic
    badwidth = total_size / duration
    print(f"From host to disk {badwidth}GB/s")


def from_disk_to_host():
    def job(i):
        data = np.load(f"/mammoth/checkpoints/{i}.npy")
   
    tic = time.time()
    pool = ThreadPoolExecutor(max_workers=total_files)
    for i in range(total_files):
        pool.submit(job, i)

    pool.shutdown(wait=True)
    toc = time.time()
    duration = toc - tic
    badwidth = total_size / duration
    print(f"From host to disk {badwidth}GB/s")

    # np.testing.assert_equal(cp.asnumpy(data[0]), mmap_files[0])

# from_device_to_ram()
# from_device_to_pinned_memory()
# from_device_to_disk()
# from_host_to_disk()
# from_disk_to_host()

file_size_gb = 10
shape = int(file_size_gb * 1e9)
dtype = np.uint8
tic = time.time()
empty_pinned(shape, dtype=dtype)
toc = time.time()
print(toc-tic)