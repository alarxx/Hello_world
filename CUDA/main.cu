#include <stdio.h>

__global__ void kernel(int * num1, int * num2, int * result){
    *result = *num1 + *num2;
}

int main(void){
    //host copies
    int     num1 = 4,
            num2 = 5,
            result;

    // device copies (GPU)
    int *p_num1, *p_num2, *p_result;

    // allocate memory on device (GPU)
    if (cudaMalloc(&p_num1, sizeof(int)) != cudaSuccess) {
        printf("Device memory allocation failure!\n");
        return 1;
    }
    cudaMalloc(&p_num2, sizeof(int));
    cudaMalloc(&p_result, sizeof(int));

    cudaMemcpy(p_num1, &num1, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(p_num2, &num2, sizeof(int), cudaMemcpyHostToDevice);

    kernel<<<1, 1>>>(p_num1, p_num2, p_result);

    // куда, откуда, кол-во байтов, Device<-Host or Host<-Device
    cudaMemcpy(&result, p_result, sizeof(int), cudaMemcpyDeviceToHost);

    printf("%d \n", result);

    cudaFree(p_num1);
    cudaFree(p_num2);
    cudaFree(p_result);
}

/*
nvcc main.cu && ./a.out
*/
