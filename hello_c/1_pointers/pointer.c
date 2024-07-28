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

- ptr - address of value to which pointer is pointing
- &ptr - address of pointer itself

- *ptr - dereferencing the pointing value
*/


/*
    Информация о значении.

    При передачи адреса значения на которое указывает pointer в функции создастся другой локальный pointer.
*/
void print_dublicate_pointer_info(char * ptr){
    printf("Dublicate ptr info:\n");
    printf("ptr Value: %c, \n", *ptr);
    printf("Address pointed by ptr: %p, \n", ptr);
    printf("Address of ptr itself: %p, \n", &ptr);
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
void print_pointer_info(char ** ptr){
    printf("Address of new pointer to ptr: %p, \n", &ptr);

    printf("ptr Value: %c, \n", **ptr);
    printf("Address pointed by original ptr: %p, \n", *ptr);
    printf("Address of original ptr: %p, \n", ptr); // &*ptr
    printf("sizeof(ptr): %lu \n\n", sizeof(ptr)); // Здесь я хз, кажется любой Pointer весит 8 байт
}


void print_variable_info(char var){
    printf("var Value: %c, \n", var);
    printf("Address of var itself: %p, \n", &var);
    printf("sizeof(var): %lu \n\n", sizeof(var));
}


int main() {

    printf("sizeof(short): %lu \n", sizeof(short));
    printf("sizeof(int): %lu \n", sizeof(int));
    printf("sizeof(long): %lu \n", sizeof(long));
    printf("sizeof(long long): %lu \n", sizeof(long long));
    printf("--------\n\n");

    printf("sizeof(char): %lu \n", sizeof(char));
    printf("sizeof(char *): %lu \n\n", sizeof(char *)); // Every pointer allocates 8 bytes
    printf("--------\n\n");


    // Выделяем память для переменной типа char
    char * ptr = malloc(sizeof(char));
    if (ptr == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    /*
        При выполнении обращайте внимание на Address of ptr itself
    */
    print_dublicate_pointer_info(ptr);
    /**
        Стоит заметить, что есть разница между передачей ptr и &ptr

        - ptr - address of value to which pointer is pointing
        - &ptr - address of pointer itself

        При передачи адреса значения на которое указывает pointer в функции создастся другой локальный pointer.

        Поэтому дальше я буду передавать адрес самого pointer-а, чтобы получать именно его информацию.
    */

    print_pointer_info(&ptr);

    printf("Our &ptr info:\n");

    printf("ptr Value: %c, \n", *ptr);
    printf("Address pointed by ptr: %p, \n", ptr);
    printf("Address of ptr itself: %p, \n", &ptr);
    printf("sizeof(ptr): %lu \n\n", sizeof(ptr));

    printf("--------\n\n");


    char a = 'A';

    print_variable_info(a);

    // Присваиваем значение выделенной памяти
    *(ptr) = a;
    // Вызовет Memory Leak
    // ptr = &a;

    print_pointer_info(&ptr);


    printf("--------\n\n");


    // Освобождаем выделенную память
    free(ptr);

    // После освобождения указатель ptr все еще хранит старый адрес, но эта память больше не доступна
    ptr = NULL; // Устанавливаем указатель в NULL, чтобы избежать висячего указателя
    /**
        Dangling Pointer - использования указателя после освобождения памяти на который он указывает.
    */

    print_variable_info(a);

    if(ptr == NULL){
        printf("ptr is NULL \n"); // nil
    }
    else { //Never will be executed
        printf("ptr Value: %c, \n", *ptr);
    }
    printf("Address pointed by ptr: %p, \n", ptr);
    printf("Address of ptr itself: %p, \n", &ptr);
    printf("sizeof(ptr): %lu \n\n", sizeof(ptr));


    return 0;
}

