vt::matmul
=======================

.. toctree::
   :maxdepth: 1

.. doxygenfunction:: vt::matmul(Tensor<T, 1>& tensor1, Tensor<T, 1>& tensor2, cublasHandle_t handle = cuda::CuBLAS::get_instance().get_handle())
.. doxygenfunction:: vt::matmul(Tensor<T, 2>& tensor1, Tensor<T, 2>& tensor2, cublasHandle_t handle = cuda::CuBLAS::get_instance().get_handle())
.. doxygenfunction:: vt::matmul(Tensor<T, N>& tensor1, Tensor<T, N>& tensor2, cublasHandle_t handle = cuda::CuBLAS::get_instance().get_handle())

