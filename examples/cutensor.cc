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
    auto tensor = vt::arange(12).reshape(2, 6);

    auto x = vt::vander(tensor, 2);


    vt::print(x);





    return 0;
}