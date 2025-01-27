
#include <iostream>
#include <vector>
#include <lib/core/mempool.hpp>
#include <lib/vtensor.hpp>

int main() {
    using namespace vt;

    size_t num_points = 1e8;
    auto timer = time::GPUTimer();

    auto x = random::rand(num_points);
    auto y = random::rand(num_points);

    timer.start();
    auto dis = sqrt(power(x, 2.0f) + power(y, 2.0f));
    auto inside = sum(dis < 1.0f);
    auto pi_estimate = double(asvector(inside)[0]) / num_points * 4;
    auto elapse = timer.stop();

    std::cout << "Pi estimate: " << pi_estimate << std::endl;
    std::cout << "Elapsed time: " << elapse << " ms" << std::endl;

    auto points = random::rand(2, num_points);
    timer.start();
    dis = sqrt(sum(power(points, 2.0f), 0));
    inside = sum(dis < 1.0f);
    pi_estimate = double(asvector(inside)[0]) / num_points * 4;
    elapse = timer.stop();

    std::cout << "Pi estimate: " << pi_estimate << std::endl;
    std::cout << "Elapsed time: " << elapse << " ms" << std::endl;
}