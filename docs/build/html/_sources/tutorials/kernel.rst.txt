GPU Kernels
==============

There are two ways for VTensor to launch GPU kernels.

CuTensor (Recommended)
------------
CuTensor, a device class, integrates seamlessly with VTensor for GPU kernel. It performs index calculations using the tensor's strides, shapes, and initial index. There are two methods for element access within a kernel:

- The `()` operator enables access via a multi-dimensional index.
- The `[]` operator allows access through a linear index, adhering to row-major order.

.. code-block:: cpp

    #include <iostream>
    #include <vtensor/lib/vtensor.hpp>

    __global__ void kernel1(vt::CuTensor<float, 2> tensor) {
        int col = threadIdx.x;
        int row = threadIdx.y;
        float value = tensor(row, col);
        printf("tensor[%d, %d] = %f\n", row, col, value);
    }

    __global__ void kernel2(vt::CuTensor<float, 2> tensor) { 
        tensor[threadIdx.x] += 1; 
    }

    int main() {
        auto tensor = vt::arange(12).reshape(4, 3);
        dim3 tpb(3, 4);
        kernel1<<<1, tpb>>>(tensor);
        cudaDeviceSynchronize();
        kernel2<<<1, 12>>>(tensor);
        cudaDeviceSynchronize();
        vt::print(tensor);

        return 0;
    }

Raw pointer
------------
Users could still use the raw pointer to access the tensor's data within a kernel. However, users need to ensure the tensor's memory is contiguous.

.. code-block:: cpp

    #include <iostream>
    #include <vtensor/lib/vtensor.hpp>

  
    __global__ void kernel(float* tensor) { 
        tensor[threadIdx.x] += 1; 
    }

    int main() {
        auto tensor = vt::arange(12).reshape(4, 3);
        kernel<<<1, 12>>>(tensor.raw_ptr());
        cudaDeviceSynchronize();
        return 0;
    }
