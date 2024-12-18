#include <iostream>
#include <lib/vtensor.hpp>

__global__ void kernel1(vt::CuTensor<float, 2> tensor) {
    int col = threadIdx.x;
    int row = threadIdx.y;
    float value = tensor(row, col);
    printf("tensor[%d, %d] = %f\n", row, col, value);
}

__global__ void kernel2(vt::CuTensor<float, 2> tensor) { tensor[threadIdx.x] += 1; }

int main() {


    auto tensor = vt::arange(12, vt::Order::F).reshape(4, 3);
    // dim3 tpb(3, 4);
    // kernel1<<<1, tpb>>>(tensor);
    // cudaDeviceSynchronize();

    auto tensor3 = tensor({1, 3, 1}, {0, 3, 2});

    vt::print(tensor3);
  
    return 0;
}