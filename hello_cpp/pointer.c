#include <stdio.h>
#include <stdlib.h>

/**
1. Статическое выделение памяти
С помощью keyword static память выделяется статически.
Статически выделенная память выделяется во время компиляции/запуска программы и не может быть изменена.

2. Автоматическое выделение памяти
Для локальных переменных функции выделяется память в стеке, которая после удаляется при завершении функции.

3. Динамическое выделение памяти

- **malloc** (n * sizeof(type)) - allocates in bytes with garbage values
- **free**(pointer to allocated memory)
- **calloc**(n, sizeof(type)) - as malloc, но инициализирует массив с нулями
- **realloc**(ptr, size) - меняет выделенный размер, новая область не инициализируется


Указатели используются для указания на динамически выделеную память во время выполнения.

Объявленные указатели, как и обычные локальные переменные выделяются в автоматической стэковой области памяти.
Когда вызывается функция для её локальных переменных выделяется память в Stack-е, локальные переменные освобождаются при завершении выполнения функции.

Pointer-ы указываются так:
    type * name = address;

`*` - is called "asterics", but in A* algorithm it's called "star"
'&' - "ampersand"

- ptr - address of value to which pointer is pointing
- &ptr - address of pointer itself

- *ptr - dereferencing the pointing value
*/


/*
    Информация о значении.

    При передачи адреса значения на которое указывает pointer в функции создастся другой локальный pointer.
*/
void print_dublicate_pointer_info(int * ptr){
    printf("Dublicate ptr info:\n");
    printf("ptr Value: %i, \n", *ptr);
    printf("Address pointed by ptr: %p, \n", ptr);
    printf("Address of ptr itself: %p, \n", &ptr); // not the same
    printf("sizeof(ptr): %lu \n\n", sizeof(ptr));
}


/*
    Информация об указателе.
    Реализуется с помощью указателя на указатель.


    Указатель на указатель ptr хранит адрес.

    - ptr (&*ptr) - адрес оригинального указателя
    - &ptr - адрес нового созданного указателя на указатель

    - *ptr - оригинальный указатель, при доступе вернет адрес variable на который указывает этот оригинальный указатель
    - **ptr - dereferencing, basically
*/
void print_pointer_info(int ** ptr){
    printf("Address of new pointer to ptr: %p, \n", &ptr);

    printf("ptr Value (address of original ptr): %p, \n", ptr); // &*ptr
    printf("Address pointed by original ptr: %p, \n", *ptr);
    printf("sizeof(ptr): %lu \n\n", sizeof(ptr)); // Здесь я хз, кажется любой Pointer весит 8 байт
}


int main() {

    int myvar = 5;
    printf("myvar: %i \n", myvar);
    printf("&myvar: %p \n", &myvar);

    int * ptr = &myvar;
    printf("ptr: %p \n", ptr);
    printf("&ptr: %p \n", &ptr);
    printf("*ptr: %li \n", *ptr);

    int ** ptr2 = &ptr;
    printf("ptr2: %p \n", ptr2);
    printf("*(&ptr2): %p \n", *(&ptr2) );
    printf("&ptr2: %p \n", &ptr2);

    printf("*ptr2 (ptr or &myvar): %p \n", *ptr2);
    printf("**ptr2 (*ptr or myvar): %li \n", **ptr2);


    print_dublicate_pointer_info(ptr);
    print_pointer_info(&ptr);


    return 0;
}

