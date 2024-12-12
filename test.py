import numpy as np
import cupy as cp
import time
from cupyx import empty_pinned

tic = time.time()
p1 = empty_pinned((1000000000), dtype=cp.float32)
p2 = empty_pinned((1000000000), dtype=cp.float32)
toc = time.time()
# print(toc - tic)

t1 = cp.arange(1000000000, dtype=cp.float32)
t2 = cp.arange(1000000000, dtype=cp.float32)

# tic = time.time()
t1.get(out=p1)
t2.get(out=p2)
cp.cuda.runtime.deviceSynchronize()
# toc = time.time()
# print(toc - tic)
# tic = time.time()
# cp.save("/home/stsui/Documents/checkpoints/0_0.npy", t1)
# cp.save("/home/stsui/Documents/checkpoints/1_1.npy", t2)
# cp.cuda.runtime.deviceSynchronize()
# toc = time.time()
# print(toc - tic)

tic = time.time()
# n1 = np.save("/home/stsui/Documents/checkpoints/0_0.npy", p1)
# n2 = np.save("/home/stsui/Documents/checkpoints/1_1.npy", p2)
n1 = np.load("/home/stsui/Documents/checkpoints/0_0.npy")
n2 = np.load("/home/stsui/Documents/checkpoints/1_1.npy")
toc = time.time()
print(toc - tic)

