#include <stdlib.h>
#include <stdio.h>

// Here we don't use typedef
struct Shape{
    // char* (*print)(void*);
    void (*print)(void*); // По сути это указатель на функцию
};

typedef struct {
    struct Shape base;
    int radius;
    int arr[10];
} Circle;

void Circle_print(void* self) {
    Circle* circle = (Circle*)self;
    printf("Circle with radius %d\n", circle->radius);
}

int main() {
    // We can initialize Shape like
    // struct Shape shape = {Circle_print};

    // But let's initialize Shape hard way
    struct Shape * shape = (struct Shape *) malloc(sizeof(struct Shape));
    (*shape).print = Circle_print;
    // shape->print = Circle_print;

    Circle circle = {*shape, 12}; // Инициализация структуры с указателем на функцию
    circle.base.print(&circle); // Вызов функции через указатель

    printf("shape %p\n", shape);
    printf("shape %p\n", *shape); // Я не знаю почему это pointer, но суть в том, что указатель не может иметь одно и то же dereference значение после free
    free(shape);
    printf("shape %p\n", shape);
    printf("shape %p\n", *shape);
    shape = NULL; // Prevent Dangling Pointer
    printf("shape %p\n", shape);

    return 0;
}
