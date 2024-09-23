## VTensor

VTensor, a C++ library, facilitates tensor manipulation on GPUs, emulating the python-numpy style for ease of use. 
It leverages RMM (RAPIDS Memory Manager) for efficient device memory management. 
The library integrates CuRand, CuBLAS, and CuSolver to support a wide range of operations, including mathematics, linear algebra, and random number generation.
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
- Support Solvers (Cholesky, QR etc.)
- Support GPUDirect
- Support more matrix operations
- Support Sparse martix with CuSparse
