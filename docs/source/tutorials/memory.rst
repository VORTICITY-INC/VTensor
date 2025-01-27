Memory Management
==============

User could leverage the `RMM device memory pool <https://github.com/rapidsai/rmm>`_ for memory allocation. Here are some ways to manage memory allocation in your program:

1. **Global Memory Pool**: This approach involves configuring the compiler options to set `-DPOOL_SIZE` to a specific percentage of the total GPU memory. An example configuration in a Bazel BUILD file demonstrates this method. Unless altered, this setting reserves 50% of each GPU device's total memory for the global memory pool. Memory allocation logs are saved to `/tmp/vtensor/memory.log`.

.. code-block:: python

   cc_binary(
       name = "my_program",
       srcs = ["my_program.cc"],
       copts = ["-DPOOL_SIZE=90", "-DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE"],
       deps = [
        "//lib:vtensor",
        "//lib/core:mempool",
       ],
   )

.. code-block:: cpp

    #include "lib/core/mempool.hpp"
    #include "lib/vtensor.hpp"

   int main(){
      // Make sure you include mempool.hpp in your program. A global memory pool is initialized for every GPU device.
   }


2. **Custom Memory Pool**: Alternatively, a custom memory pool can be instantiated by specifying its size and the log file's path. This pool is dedicated to the GPU device in use. 

.. code-block:: python

   cc_binary(
       name = "my_program",
       srcs = ["my_program.cc"],
       copts = ["-DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE"],
       deps = [
        "//lib:vtensor",
        "//lib/core:rmm_utils",
    ],
   )

.. code-block:: cpp

   #include "lib/core/rmm_utils.hpp"

   int main(){
       auto pool = vt::Mempool(50, "/tmp/vtensor/memory.log"); // Current GPU device's memory pool
   }

.. code-block:: cpp

   #include "lib/core/rmm_utils.hpp"

   int main(){
       auto pool = vt::GlobalMempool(50); // Initialize global memory pool for every GPU device
   }
