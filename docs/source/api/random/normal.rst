vt::normal
=======================

.. toctree::
   :maxdepth: 1

.. doxygenfunction:: vt::random::normal(Shape<N> shape, T mean = T{0.0}, T stddev = T{1.0}, const Order order = Order::C, curandGenerator_t gen = cuda::CuRand::get_instance().get_handle())
.. doxygenfunction:: vt::random::normal(size_t m, T mean = T{0.0}, T stddev = T{1.0}, const Order order = Order::C, curandGenerator_t gen = cuda::CuRand::get_instance().get_handle())
.. doxygenfunction:: vt::random::normal(size_t m, size_t n, T mean = T{0.0}, T stddev = T{1.0}, const Order order = Order::C, curandGenerator_t gen = cuda::CuRand::get_instance().get_handle())
