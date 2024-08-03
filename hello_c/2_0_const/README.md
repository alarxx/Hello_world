# C Programming Language - const

keyword : `const`.

Константные переменные read-only, их нельзя изменять.

| Основные моменты:
```c
int a = 10;

int * ptr = a;

// const значение *ptr
const int * ptr = a;

// const значение &ptr
int * const ptr = a;

// полный запрет на изменение всего
const int * const ptr = a;

// Взлом constant-ы LOLL
const int CONST = 999;
int * ptr = &CONST;
*cptr = 123;
```

| Example:
```c
 #include <stdio.h>

int main() {
    const int CONST = 999;
    printf("CONST = %i \n", CONST);
    // CONST = 888; error: assignment of read-only variable ‘CONST’

    int x = 10;
    int y = 20;

    // Константный указатель на изменяемые данные
    int * const ptr1 = &x;
    *ptr1 = 15; // Разрешено
    // ptr1 = &y; // Ошибка: нельзя изменить указатель

    // Изменяемый указатель на константные данные
    const int *ptr2 = &x;
    // *ptr2 = 15; // Ошибка: нельзя изменить данные через указатель
    ptr2 = &y; // Разрешено

    // Константный указатель на константные данные
    const int *const ptr3 = &x;
    // *ptr3 = 15; // Ошибка: нельзя изменить данные
    // ptr3 = &y; // Ошибка: нельзя изменить указатель


    // Взлом constant-ы LOLL
    int * cptr = &CONST;
    *cptr = 123;
    printf("cptr = %i \n", *cptr);
    printf("CONST = %i \n", CONST);

    return 0;
}
```
