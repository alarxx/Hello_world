#include <stdio.h>
#include <stdlib.h>

typedef int binary_op(int, int);
typedef int (* binary_op_ptr)(int, int);

int sum(int x, int y){
    // sizeof sum не зависит от размера кода и всегда 1 байт?
    // int a = 1;
    // int b = 2;
    // int c = a + b;
    // int d = c + b;
    // int e = d + b;
    // int f = e + b;
    // int g = f + b;
    // int h = g + b;
    // int i = h + b;
    // int j = i + b;
    // printf("j = %i \n", j);
    return x + y;
}

int sub(int x, int y){
    return x - y;

}

int operation_straight(int (*op)(int, int), int a, int b){
    return (*op)(a, b);
    // return op(a, b);
}

int operation(binary_op op, int a, int b){
    return op(a, b);
}

int operation_wptr(binary_op_ptr op_ptr, int a, int b){
    return op_ptr(a, b);
}

// int main(int argc, char *argv[])
int main(void){

    printf("sizeof binary_op = %lu \n", sizeof(binary_op));
    printf("sizeof binary_op_ptr = %lu \n", sizeof(binary_op_ptr));
    printf("sizeof sum = %lu \n", sizeof(sum));
    printf("sizeof &sum = %lu \n", sizeof(&sum));

    printf("sum result = %d \n", operation_straight(&sum, 10, 5));
    // result = 15
    printf("sub result = %d \n", operation(&sub, 10, 5));
    // result = 5

    int a = (*&sum)(1, 2);
    printf("a = %i \n", a);

    return 0;

}
