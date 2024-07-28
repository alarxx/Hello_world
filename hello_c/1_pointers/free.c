#include <stdio.h>
#include <stdlib.h>

int main(){
    int * ptr = malloc(8);

    printf("ptr = %p \n", ptr); // 0x01234ABC
    printf("&ptr = %p \n", &ptr); // 0x56789DEF

    free(ptr);
    // ptr = NULL;
    printf("ptr = %p \n", ptr); // 0x01234ABC still
    printf("&ptr = %p \n", &ptr); // 0x56789DEF
}
