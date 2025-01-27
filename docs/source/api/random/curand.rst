vt::cuda:CuRand
=======================

.. toctree::
   :maxdepth: 1

.. doxygenstruct:: vt::cuda::Default
   :members:

.. doxygenstruct:: vt::cuda::XORWOW
   :members:

.. doxygenstruct:: vt::cuda::MMRG32K3A
   :members:

.. doxygenstruct:: vt::cuda::MTGP32
   :members:

.. doxygenstruct:: vt::cuda::MT19937
   :members:

.. doxygenstruct:: vt::cuda::PHILOX4_32_10
   :members:

.. doxygenstruct:: vt::cuda::SOBOL32
   :members:

.. doxygenstruct:: vt::cuda::SCRAMBLED_SOBOL32
   :members:

.. doxygenstruct:: vt::cuda::SOBOL64
   :members:

.. doxygenstruct:: vt::cuda::SCRAMBLED_SOBOL64
   :members:

.. doxygenstruct:: vt::cuda::RNGTypeTraits
   :members:

.. doxygenfunction:: vt::cuda::check_curand_status(curandStatus_t status, const std::string& message)

.. doxygenstruct:: vt::cuda::CuRandHandleDeleter
   :members:

.. doxygenfunction:: vt::cuda::create_curand_handle(size_t dim = 1)

.. doxygenclass:: vt::cuda::CuRand
   :members:

.. doxygenfunction:: vt::cuda::set_seed(size_t seed, curandGenerator_t gen = curand.get_handle())
.. doxygenfunction:: vt::cuda::set_offset(size_t offset, curandGenerator_t gen = curand.get_handle())
