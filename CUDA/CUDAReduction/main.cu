#include <iostream> // std::cout
#include <stdio.h> // printf
#include <cuda_runtime.h>
#include <cstdlib> // malloc
#include <cassert> // assert
#include <chrono>

#include "dli.hpp"

__global__ void test_kernel(){
    // `std::cout` не существует на device-е, для вывода только `printf`
    printf("Execution space: %s\n", execution_space());
}

/*

// tree-based reduction, but with bank conflicts
// stride=1 will result in only 128/32=4 bank conflicts
// stride of a*32 will result in all bank conflicts
for (int stride = 1; stride < blockDim.x; stride *= 2) {
    int i = 2 * stride * tid;     // левый индекс пары
    if (i < blockDim.x) {
        sdata[i] += sdata[i + stride];
    }
    __syncthreads();
}

*/
__global__ void reduce_sum(
    const float* input,
    float* block_sums,
    int n
) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // write into shared memory
    if (gid < n)
        sdata[tid] = input[gid];
    else
        sdata[tid] = 0.0f;
    __syncthreads();

    // tree-based reduction
    for (int s = blockDim.x / 2; s > 0; s /= 2) { // 4, 2, 1
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // partial sum
    if (tid == 0) {
        block_sums[blockIdx.x] = sdata[0];
    }
}



int main(){
    test_kernel<<<1, 1, 0, 0>>>();

    int n = 2 * 1 << 20;
    size_t bytes = n * sizeof(float);

    float* h_input = new float[n];
    for (int i = 0; i < n; ++i) {
        h_input[i] = 1.0f;
    }



    auto t0 = std::chrono::high_resolution_clock::now();
    float cpu_sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        cpu_sum += h_input[i];
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double secondsh = std::chrono::duration<double>(t1 - t0).count();
    double msh = secondsh * 1000;
    printf("CPU time: %.10f ms\n", msh );



    float * d_input;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));

    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    // std::cout << blocks << std::endl; // 8192

    float * d_block_sums;
    CUDA_CHECK(cudaMalloc(&d_block_sums, blocks * sizeof(float)));

    // warmup
    {
        reduce_sum<<<blocks, threads, threads * sizeof(float), 0>>>(
            d_input,
            d_block_sums,
            n
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, 0));
    reduce_sum<<<blocks, threads, threads * sizeof(float), 0>>>(
        d_input,
        d_block_sums,
        n
    );
    CUDA_CHECK(cudaGetLastError());
    // CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));


    float* h_block_sums = new float[blocks];
    CUDA_CHECK(cudaMemcpy(
        h_block_sums,
        d_block_sums,
        blocks * sizeof(float),
        cudaMemcpyDeviceToHost
    ));


    t0 = std::chrono::high_resolution_clock::now();
    float sum = 0.0f;
    for (int i = 0; i < blocks; ++i) {
        sum += h_block_sums[i];
    }
    t1 = std::chrono::high_resolution_clock::now();
    double secondsk = std::chrono::duration<double>(t1 - t0).count();
    double msk = ms + secondsk * 1000;
    std::cout << "CUDA Kernel time: " << msk << " ms" << std::endl;
    std::cout << (msh / msk) << " times faster!" << std::endl;



    std::cout << "GPU reduction result = " << (int) sum << std::endl;
    std::cout << "Expected = " << n << std::endl;

    assert(sum == n);



    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    delete[] h_input;
    delete[] h_block_sums;
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_block_sums));
}
