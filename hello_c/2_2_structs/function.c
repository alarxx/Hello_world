
#include <stdio.h>
#include <stdlib.h>


typedef struct {
    void (* print) (void *);
} Shape;


typedef struct {
    Shape base;
    int radius;
} Circle;


void Circle_print(void * self) {
    Circle * circle = (Circle *) self;
    printf("Circle->radius %d\n", circle->radius);
    printf("(*Circle).radius %d\n", (*circle).radius);
}

void function(void * param){
    return;
}

int main() {
    printf("1 > 2 = %i - FALSE \n", (1 > 2)); // 0 - FALSE
    printf("1 < 2 = %i - TRUE \n", (1 < 2)); // 1 - TRUE

    printf("sizeof function = %lu \n", sizeof(function));
    printf("sizeof &function = %lu \n", sizeof(&function));

    printf("--------------------------\n\n");

    // Инициализация структуры с указателем на функцию
    Circle circle = {
        .base={Circle_print},
        .radius=10
    };

    circle.base.print(&circle); // Вызов функции через указатель

    return 0;
}
