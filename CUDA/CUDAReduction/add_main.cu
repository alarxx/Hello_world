#include <iostream> // std::cout
#include <stdio.h> // printf
#include <cuda_runtime.h>
#include <cstdlib> // malloc
#include <cassert> // assert

#include "dli.hpp"

__global__ void test_kernel(){
    // `std::cout` не существует на device-е, для вывода только `printf`
    printf("Execution space: %s\n", execution_space());
}


__global__ void vector_add(
    const float* a,
    const float* b,
    float* out,
    int n
){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < n){
        out[tid] = a[tid] + b[tid];
    }
}


int main(){
    std::cout << execution_space() << std::endl;

    test_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    // manual cuda error check
    cudaError_t err = cudaGetLastError();
    printf("CUDA error: %s\n", cudaGetErrorString(err));


    // 2^20 = 1 048 576 ; x4 Bytes = 4MB
    // 2^30 = 1 073 741 824; x4 Bytes = 4GB
    int n = 1 << 20;
    const size_t bytes = n * sizeof(float);

    // float * a = (float *) std::malloc(bytes);
    float * a = new float[n];
    float * b = new float[n];
    float * out = new float[n];

    for(int i = 0; i < n; i++){
        a[i] = (float) i;
        b[i] = (float) (2 * i);
    }

    float   * d_a,
            * d_b,
            * d_out;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // kernel parameters
    int threads_per_block = 1024; // 32 warps, 8 clocksteps (128 datapaths)
    int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;
    assert(threads_per_block <= 1024 && blocks_per_grid <= 1024);

    // warmup
    vector_add<<<blocks_per_grid, threads_per_block, 0, 0>>>(d_a, d_b, d_out, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // kernel launch
    cudaEventRecord(start, 0);
    for (int i = 0; i < 4000; ++i) {
        vector_add<<<blocks_per_grid, threads_per_block, 0, 0>>>(d_a, d_b, d_out, n);
    }
    CUDA_CHECK(cudaGetLastError());
    // CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop); // CPU

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    printf("time: %.10f s\n", get_seconds(ms));

    CUDA_CHECK(cudaMemcpy(out, d_out, bytes, cudaMemcpyDeviceToHost));

    // Verification
    std::cout << "i + 2 * i" <<  std::endl;
    for (int i = 0; i < n; ++i) {
        if(i < 5){
            printf("%i + 2 * %i: %f \n", i, i, out[i]);
        }
        assert(out[i] == a[i] + b[i]);
    }

    // std::free(a);
    delete[] a;
    delete[] b;
    delete[] out;

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_out));

    return 0;
}

/*
nvcc add_main.cu -o main.out -arch=sm_75 --use_fast_math -Xcompiler "-Wall -Wextra" && ./main.out
*/
