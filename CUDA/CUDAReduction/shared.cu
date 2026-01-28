#include <cstdio>
#include "dli.hpp"

__global__ void show_shared_size() {
    extern __shared__ float sdata[];

    if (threadIdx.x == 0) {
        printf(
            "Block %d: shared start addr = %p\n",
            blockIdx.x,
            sdata
        );
    }
}

int main() {
    int threads = 128;

    // 128 float = 512 bytes
    show_shared_size<<<1, threads, threads * sizeof(float), 0>>>();

    // 256 float = 1024 bytes
    show_shared_size<<<1, threads, 10 * threads * sizeof(float), 0>>>();

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaDeviceSynchronize();

    return 0;
}
