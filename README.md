## VTensor

VTensor, a C++ library, facilitates tensor manipulation on GPUs, emulating the python-numpy style for ease of use. 
It leverages RMM (RAPIDS Memory Manager) for efficient device memory management. 
The library integrates CuRand, CuBLAS, and CuSolver to support a wide range of operations, including mathematics, linear algebra, and random number generation. It also supports transferring the device memory to/from host memory with xtensor project (https://github.com/xtensor-stack/xtensor).
Please visit our website https://vorticity-inc.github.io/VTensor/ for more information!

### Test
```sh
bazel test ...
```

### Formatter
```sh
find lib  -name '*.hpp' |  xargs clang-format -i -style=file:clang.yaml 
```

### Update docs
```sh
doxygen docs/Doxyfile.in
sphinx-build -b html docs/source docs/build/html
```

### Future updates
- The current tensor supports both "C" and "F" memory layouts. However, mixed order operations are not prohibited, and the "F" order does not fully support linear algebra functions.
- Support more matrix operations
- Support Sparse martix with CuSparse
