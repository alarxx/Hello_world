/* Polymorphism

 In C you can't create functions with the same name.
 But you can in C++ (Polymorphism).

 C:
 $ gcc fun.c
 > Error

 C++:
 $ g++ fun.c
 > Ok

 */
#include <stdio.h> // There is still C headers in C++, or <cstdio>

int     fun(int     x){ return x; }
double  fun(double  x){ return x; }
float   fun(float   x){ return x; }

int main(){
    printf("int:    %d \n", fun(1   ));
    printf("double: %f \n", fun(1.0 ));
    printf("float:  %f \n", fun(1.0f));
}
