Memory Management
==============

VTensor leverages the `RMM device memory pool <https://github.com/rapidsai/rmm>`_ for memory allocation by default, offering two strategies for GPU memory management:

1. **Global Memory Pool**: This approach involves configuring the compiler options to set `-DPOOL_SIZE` to a specific percentage of the total GPU memory. An example configuration in a Bazel BUILD file demonstrates this method. Unless altered, this setting reserves 50% of each GPU device's total memory for the global memory pool. Memory allocation logs are saved to `/tmp/vtensor/memory.log`.

.. code-block:: python

   cc_binary(
       name = "my_program",
       srcs = ["my_program.cc"],
       copts = ["-DPOOL_SIZE=90", "-DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE"],
       deps = ["@vtensor//:vtensor"],
   )

2. **Custom Memory Pool**: Alternatively, a custom memory pool can be instantiated by specifying its size and the log file's path. This pool is dedicated to the GPU device in use.

.. code-block:: cpp

   #include <vtensor/lib/vtensor.hpp>

   int main(){
       auto pool = vt::MemPool(50, "/tmp/vtensor/memory.log");
   }
