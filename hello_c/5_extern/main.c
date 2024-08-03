#include <stdio.h>

/*
    Объявления и использование
*/

// Compilation Error: multiple definition of `FOOBAR';
// int FOOBAR;
extern int FOOBAR;

// При объявлении функции `extern` можно опустить
int add(int a, int b);
extern int sub(int a, int b);
extern int substract(int a, int b);


int main(){

    printf("FOOBAR = %i \n", FOOBAR);

    int a = 2, b = 3;

    int c = add(a, b);
    printf("c = %i \n", c);

    c = sub(a, b);
    printf("c = %i \n", c);

    // Compilation Error: undefined reference to `substract'
    // c = substract(a, b);
    // printf("c = %i \n", c);
}
