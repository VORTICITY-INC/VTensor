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
    auto tensor = vt::arange(10).reshape(2, 5);
    auto y = tensor * 2.0f + 1.0f;

    auto x = vt::vander(tensor, 2);

    auto xinv = vt::linalg::pinv_weak(x);
    auto _y = y(vt::ellipsis, vt::newaxis);
    auto coefs = vt::matmul(xinv, _y);

    vt::print(coefs);


    // degree = 2

    // # Create the Vandermonde matrix A
    // A = np.vander(x, degree + 1)

    // # Solve for coefficients using the normal equation
    // pseudo_inv = np.linalg.inv(A.T @ A) @ A.T
    // coefficients = pseudo_inv @ y




    return 0;
}