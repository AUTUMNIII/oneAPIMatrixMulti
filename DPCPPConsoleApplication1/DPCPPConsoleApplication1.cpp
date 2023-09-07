#include <CL/sycl.hpp>
#include <iostream>

constexpr size_t N = 1024;

int main() {
    std::vector<float> matrixA(N * N, 2.0f);
    std::vector<float> matrixB(N * N, 3.0f);
    std::vector<float> matrixC(N * N, 0.0f);

    try {
        sycl::queue myQueue;
        sycl::range<2> size(N, N);

        sycl::buffer<float, 2> bufferA(matrixA.data(), size);
        sycl::buffer<float, 2> bufferB(matrixB.data(), size);
        sycl::buffer<float, 2> bufferC(matrixC.data(), size);

        myQueue.submit([&](sycl::handler& cgh) {
            auto accessorA = bufferA.get_access<sycl::access::mode::read>(cgh);
            auto accessorB = bufferB.get_access<sycl::access::mode::read>(cgh);
            auto accessorC = bufferC.get_access<sycl::access::mode::write>(cgh);

            cgh.parallel_for<class MatrixMultiply>(size, [=](sycl::id<2> idx) {
                float sum = 0.0f;
                for (int k = 0; k < N; ++k) {
                    sum += accessorA[idx[0]][k] * accessorB[k][idx[1]];
                }
                accessorC[idx] = sum;
                });
            });

        myQueue.wait();
    }
    catch (sycl::exception const& e) {
        std::cerr << "An exception occurred: " << e.what() << std::endl;
        return 1;
    }

    // ´òÓ¡½á¹û
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            std::cout << matrixC[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}