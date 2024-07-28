#include <stdio.h>
#include <stdlib.h>

int main(){
    int * ptr = malloc(8);

    printf("ptr = %p \n", ptr);
    printf("&ptr = %p \n", &ptr);

    free(ptr);
    // ptr = NULL;
    printf("ptr = %p \n", ptr);
    printf("&ptr = %p \n", &ptr);
}
