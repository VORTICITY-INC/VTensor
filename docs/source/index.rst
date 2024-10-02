VTensor documentation
=====================

VTensor, a C++ library, facilitates tensor manipulation on GPUs, emulating the python-numpy style for ease of use. 
It leverages RMM (RAPIDS Memory Manager) for efficient device memory management. 
The library integrates CuRand, CuBLAS, and CuSolver to support a wide range of operations, including mathematics, linear algebra, and random number generation.
Explore one of our examples for a quick overview!

.. code-block:: cpp

   #include <lib/vtensor.hpp>

   int main() {
      auto num_points = 100000;
      auto points = vt::random::rand(2, num_points); // 2D random points
      auto dist = vt::sqrt(vt::sum(vt::power(points, 2.0f), 0)); // Euclidean distance along the axis 0
      auto inside_points = vt::sum(dist < 1.0f); // Count points inside the unit circle
      auto pi_estimate = double(asvector(inside)[0]) / num_points * 4; // Copy the result from device to host and estimate pi
   }


.. toctree::
   :maxdepth: 1

   installation
   tutorials/index
   api/index
