#include <iostream> // std::cout
#include <stdio.h> // printf
#include <cuda_runtime.h>
#include <cstdlib> // malloc
#include <chrono>


double get_seconds(float ms){
    double s = ms * 0.001;
    // double microseconds = ms * 1000;
    // double ns = ms * 1000 * 1000;
    return s;
}

int main(){
    // 2^30 = 1 073 741 824; x4 Bytes = 4GB
    long n = 1 << 30;
    const size_t bytes = n * sizeof(float);
    int iters = 16;
    std::cout << "Number of floats: " << n << "; size: " << bytes << std::endl;

    auto t0 = std::chrono::high_resolution_clock::now();

    // float * a = (float *) std::malloc(bytes);
    // float * h_a = new float[n];
    float * h_a;
    cudaMallocHost(&h_a, bytes); // Pinned memory
    float * d_a;
    cudaMalloc(&d_a, bytes); // GPU

    // Initialize
    for(int i = 0; i < n; i++){
        h_a[i] = (float) i;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // // warmup
    // cudaMemcpyAsync(d_a, h_a, bytes, cudaMemcpyHostToDevice, stream);
    // cudaStreamSynchronize(stream);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, stream);

    // dst <- src
    for (int i = 0; i < iters; ++i) {
        cudaMemcpyAsync(d_a, h_a, bytes, cudaMemcpyHostToDevice, stream);
    }

    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop); // CPU

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    double seconds = get_seconds(ms);
    printf("Memcpy time: %.10f s\n", seconds);

    // Bandwidth
    double total_GB = (double) bytes * iters / 1e9;
    double bw = total_GB / seconds;
    printf("Total H2D: %.2f GB in %.3f s => %.2f GB/s\n", total_GB, seconds, bw);

    // Total
    auto t1 = std::chrono::high_resolution_clock::now();
    seconds = std::chrono::duration<double>(t1 - t0).count();
    printf("Program time: %.10f s\n", seconds);

    return 0;
}

/*
nvcc memcpy_time.cu -o main.out -arch=sm_75 --use_fast_math -Xcompiler "-Wall -Wextra" && ./main.out
*/
