#pragma once

#include <iostream> // std::cout
#include <stdio.h> // printf
#include <cuda_runtime.h>

// #ifdef DEBUG
// CUDA error checking: every CUDA API call should be wrapped
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err__ = (call); \
        if (err__ != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err__) \
            << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)
// # endif

__host__ __device__ inline const char* execution_space() {
    #ifdef __CUDA_ARCH__
        return "device";  // если компилируется для GPU
    #else
        return "host";    // если компилируется для CPU
    #endif
}

double get_seconds(float ms){
    double s = ms * 0.001;
    // double microseconds = ms * 1000;
    // double ns = ms * 1000 * 1000;
    return s;
}
