#include <stdio.h>
#include <stdlib.h>

// Static Memory Allocation
void function(){
	static int attached = 0;
	attached++;
	printf("attached = %d \n", attached);
}


int main(){
    printf("Hello, World! \n");
    printf("sizeof(int) %ld \n", sizeof(int));

    // Automatic (Stack) Memory Allocation
    int n = 5;//20000000;
    // scanf("n: %d", &n);
    // // int arr[5] = {10, 11, 2, 3, 4};

    // Dynamic Memory Allocation,
    // needs to free up memory in order of avoiding memory leak
    int *arr = malloc(n * sizeof(int)); // указатель на первый элемент массива
    printf("sizeof(arr) = %lu \n", sizeof(&arr));
    //
    for(int i=0; i<n; i++){
        if(i<2){ // 0 1
            arr[i] = i + 10;
        }
        else {
            *(arr + i) = i;
        }
    }

    printf("sizeof(arr) = %lu \n", sizeof(&arr));


    printf("*arr = %d \n", *arr); // выводим не адрес (указатель), а само значение на которое указывается

    for(int i=0; i<n; i++){
        printf("arr[%d] = %d \n", i, *(arr + i));
        // printf("arr[%d] = %d \n", i, arr[i]); // same
    }

    free(arr);
    arr = NULL; // Висячий указатель (Dangling Pointer)
    // avoid using freed pointers

    for(int i=0; i<5; i++){
        function();
    }

    return 0;
}

// gcc -Wall source.c -o source.bin
// ./source.bin
