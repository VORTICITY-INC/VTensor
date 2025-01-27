vt::linalg::inv
=======================

.. toctree::
   :maxdepth: 1

.. doxygenfunction:: vt::linalg::inv(Tensor<T, 2>& tensor, cusolverDnHandle_t handle = cuda::CuSolver::get_instance().get_handle())
.. doxygenfunction:: vt::linalg::inv(Tensor<T, N>& tensor, cublasHandle_t handle = cuda::CuBLAS::get_instance().get_handle())
