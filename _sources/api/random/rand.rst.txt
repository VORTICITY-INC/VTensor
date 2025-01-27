vt::rand
=======================

.. toctree::
   :maxdepth: 1

.. doxygenfunction:: vt::random::rand(Shape<N> shape, const Order order = Order::C, curandGenerator_t gen = cuda::CuRand::get_instance().get_handle())
.. doxygenfunction:: vt::random::rand(size_t m, const Order order = Order::C, curandGenerator_t gen = cuda::CuRand::get_instance().get_handle())
.. doxygenfunction:: vt::random::rand(size_t m, size_t n, const Order order = Order::C, curandGenerator_t gen = cuda::CuRand::get_instance().get_handle())
