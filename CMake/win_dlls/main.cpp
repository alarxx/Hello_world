#include <iostream>

int add(int a, int b);
int sub(int a, int b);

int main(){
    std::cout << "1 + 1 = " << add(1, 1) << std::endl;
    std::cout << "1 - 1 = " << sub(1, 1) << std::endl;
    return 0;
}

/*
#include <stdio.h>

int add(int a, int b);
int sub(int a, int b);

int main(){
    printf("1 + 1 = %d \n", add(1, 1));
    printf("1 - 1 = %d \n", sub(1, 1));
    return 0;
}
*/
