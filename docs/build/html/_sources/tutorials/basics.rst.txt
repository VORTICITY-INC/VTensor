Basics of VTensor
==============

Create tensors on GPU
------------
VTensor offers several generators to create tensors on a GPU device. If the user does not specify the data type, the default data type is `float`. The following code snippet demonstrates how to create tensors with different values.

.. code-block:: cpp

   #include <vtensor/lib/vtensor.hpp>

   int main() {
      auto a = vt::arange(10); // Create a tensor with values from 0 to 9
      auto b = vt::ones(10); // Create a tensor with all elements as 1
      auto c = vt::zeros(10); // Create a tensor with all elements as 0
      auto d = vt::eye(3); // Create a 3x3 identity matrix
      auto e = vt::zeros<int>(10, 10); // Create a 2D tensor with all elements as 0
      auto f = vt::ones<bool>(10, 10); // Create a 2D tensor with all elements as True
      vt::print(f); // Print the tensor
   }

Basics operations
------------
Users could also do a variety of operations on tensors, such as addition, subtraction, multiplication, and division.

.. code-block:: cpp

   #include <vtensor/lib/vtensor.hpp>

   int main() {
      auto a = vt::arange(10); // Create a tensor with values from 0 to 9
      auto b = vt::ones(10); // Create a tensor with all elements as 1
      a += 1.0f; // Add 1 to each element
      b = b + a; // Add tensor a to tensor b
      b = b * 2.0f; // Multiply each element by 2
      vt::print(b); // Print the tensor
    }

VTensor also supports reshape and slice operations without copying the data. The index `[]` operator performs slicing along the last axis and reduces the tensor's dimension by one.
The bracket `({start, stop, step})` operator performs a slice operation along the specified axes and returns a tensor with the same dimension as the original tensor. Notice that the slicing operation will make the tensor's memory non-contiguous. 
Some of the functions (e.g. CuBLAS, CuSolver) might require a contiguous memory layout. In such cases, the tensor will be copied to a contiguous memory layout.

.. code-block:: cpp

   #include <vtensor/lib/vtensor.hpp>

   int main() {
      auto a = vt::arange(12).reshape(2, 2, 3); // Create a 3D tensor.
      auto b = a[0]; // Slice along the first axis, resulting in a 2x3 tensor.
      b = a({0, 2, 1}, {0, 2, 2}); // Slice along the first and second axes, resulting in a 2x1 tensor.
      vt::print(b); // Print the tensor
    }

Linear algebra
------------

VTensor supports a wide range of mathematical and linear algebra operations, such as `sort` `sum` `matmul`. Please see the `API documentation` for more details.

.. code-block:: cpp

   #include <vtensor/lib/vtensor.hpp>

   int main() {
      auto a = vt::arange(25).reshape(5, 5); // Create a 5x5 tensor
      auto b = vt::matmul(a, a); // Matrix multiplication
      auto c = vt::sum(b, 0); // Sum along the axis 0
      vt::print(c); // Print the tensor
    }

Random number generation
------------

VTensor utilizes CuRand to generate random numbers on the GPU. A global CuRand handeler has been created to reduce the overhead of creating and destroying CuRand handlers. The default is Peudo-random XORWOW generator.

.. code-block:: cpp

    #include <vtensor/lib/vtensor.hpp>
    
    int main() {
        auto a = vt::random::rand(10); // Create a tensor with random values
        auto b = vt::random::randn(10); // Create a tensor with random values from a normal distribution
        vt::print(b); // Print the tensor
     }

Users could create a new CuRand handler, For example, to create a new CuRand handler with the ScrambledSobol32 for quasi-random number generation. The handeler is a unique pointer with CuRandHandleDeleter as the custom deleter.

.. code-block:: cpp

    #include <vtensor/lib/vtensor.hpp>
    
    int main() {
        auto dim = 10;
        auto gen = vt::cuda::create_curand_handle<vt::cuda::SCRAMBLED_SOBOL32>(dim); // Create a new CuRand handler
        auto a = vt::random::rand(10, *gen.get()); // Create a tensor with qausi-random values
        auto b = vt::random::randn(10, *gen.get()); // Create a tensor with qausi-random values from a normal distribution
        vt::print(b); // Print the tensor
     }
