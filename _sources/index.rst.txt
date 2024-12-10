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


Comparison between NumPy and VTensor
----------------------

.. list-table:: Initializers
   :header-rows: 1

   * - NumPy
     - VTensor
   * - np.ones
     - vt::ones
   * - np.zeros
     - vt::zeros
   * - np.arange
     - vt::arange
   * - np.eye
     - vt::eye

.. list-table:: Slicing
   :header-rows: 1

   * - NumPy
     - VTensor
   * - arr[1:3]
     - tensor({1, 3})
   * - arr[..., 1:3]
     - tensor(vt::ellipsis, {1, 3})
   * - arr[..., np.newaxis]
     - tensor(vt::ellipsis, vt::newaxis)
   * - arr[np.newaxis, ...]
     - tensor(vt::newaxis, vt::ellipsis)
   * - arr[0]
     - tensor[0]
   * - arr[None, None, ...]
     - vt::expand_dims_lhs<T, N, 2>(tensor)
   * - arr[..., None, None]
     - vt::expand_dims_rhs<T, N, 2>(tensor)

.. list-table:: Broadcasting
   :header-rows: 1

   * - NumPy
     - VTensor
   * - np.broadcast
     - vt::broadcast
   * - np.broadcast_to
     - vt::broadcast_to
   * - Broadcasting operations (+, - etc.)
     - Broadcasting operations (+, - etc.)

.. list-table:: Assignment
   :header-rows: 1

   * - NumPy
     - VTensor
   * - arr[1:3] = 0
     - tensor({1, 3}) = 0
   * - arr[1:3] = arr1
     - tensor({1, 3}) = tensor1
   * - arr[1:3] = arr2[1:3]
     - tensor({1, 3}) = tensor2({1, 3})
   * - arr[arr > 0] = 0
     - tensor[tensor > 0] = 0

.. list-table:: Rearrange
   :header-rows: 1

   * - NumPy
     - VTensor
   * - arr.reshape
     - tensor.reshape
   * - arr.astype
     - tensor.astype
   * - np.diag
     - vt::diag
   * - np.tri
     - vt::tri
   * - np.tril
     - vt::tril
   * - np.triu
     - vt::triu
   * - np.transpose
     - vt::transpose
   * - np.swapaxes
     - vt::swapaxes
   * - np.moveaxis
     - vt::moveaxis

.. list-table:: Logical
   :header-rows: 1

   * - NumPy
     - VTensor
   * - np.where
     - vt::where
   * - np.any
     - vt::any
   * - np.all
     - vt::all

.. list-table:: Minimum, Maximum, Sorting
   :header-rows: 1

   * - NumPy
     - VTensor
   * - np.max
     - vt::max
   * - np.max(a, axis=-1)
     - vt::max(a, axis=-1)
   * - np.min
     - vt::min
   * - np.min(a, axis=-1)
     - vt::min(a, axis=-1)
   * - np.sum
     - vt::sum
   * - np.sum(a, axis=-1)
     - vt::sum(a, axis=-1)
   * - np.mean
     - vt::mean
   * - np.mean(a, axis=-1)
     - vt::mean(a, axis=-1)
   * - np.maximum
     - vt::maximum
   * - np.minimum
     - vt::minimum
   * - np.sort
     - vt::sort
   * - np.sort(a, axis=-1)
     - vt::sort(a, axis=-1)

.. list-table:: Mathematics
   :header-rows: 1

   * - NumPy
     - VTensor
   * - np.exp
     - vt::exp
   * - np.power
     - vt::power
   * - np.sqrt
     - vt::sqrt
   * - np.vander
     - vt::vander

.. list-table:: Random
   :header-rows: 1

   * - NumPy
     - VTensor
   * - np.random.rand
     - vt::random::rand
   * - np.random.normal
     - vt::random::normal

.. list-table:: Linear Algebra
   :header-rows: 1

   * - NumPy
     - VTensor
   * - np.matmul
     - vt::matmul
   * - np.dot
     - vt::dot
   * - np.linalg.cholesky
     - vt::linalg::cholesky
   * - np.linalg.svd
     - vt::linalg::svd
   * - np.linalg.inv
     - vt::linalg::inv
   * - np.linalg.pinv
     - vt::linalg::pinv

.. list-table:: Memory/IO
   :header-rows: 1

   * - NumPy
     - VTensor
   * - np.ascontiguousarray
     - vt::ascontiguoustensor
   * - np.asfortranarray
     - vt::asfortrantensor
   * - N/A
     - vt::asvector
   * - N/A
     - vt::astensor
   * - N/A
     - vt::asxarray
   * - np.copy
     - vt::copy
   * - np.save
     - vt::save
   * - np.load
     - vt::load

.. toctree::
   :maxdepth: 1

   installation
   tutorials/index
   api/index
