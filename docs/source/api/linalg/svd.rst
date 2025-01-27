vt::linalg::svd
=======================

.. toctree::
   :maxdepth: 1

.. doxygenfunction:: vt::linalg::svd(Tensor<T, 2>& tensor, bool full_matrices = true, bool compute_uv = true, cusolverDnHandle_t handle = cuda::CuSolver::get_instance().get_handle())
.. doxygenfunction:: vt::linalg::svd(Tensor<T, N>& tensor, bool full_matrices = true, bool compute_uv = true, cusolverDnHandle_t handle = cuda::CuSolver::get_instance().get_handle())
