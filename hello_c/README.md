# Overview

---

# Memory Allocation

In `./1_pointers`

## Automatic Memory Allocation in Stack

**Автоматическое выделение памяти**

Для локальных переменных функции выделяется память в стеке, которая после удаляется при завершении функции.

```c
void function(){
    int a = 10;
}
```

## Static Memory Allocation

**Статическое выделение памяти**  

С помощью keyword static память выделяется статически.
Статически выделенная память выделяется во время компиляции/запуска программы и не может быть изменена.

keyword `static`
```c
void function(){
    static int a = 10;
}
```

## Dynamical Memory Allocation and Pointers

**Динамическое выделение памяти**

```c
#include <stdlib.h>
```

- **malloc** (n * sizeof(type)) - allocates in bytes with garbage values
- **free**(pointer to allocated memory)
- **calloc**(n, sizeof(type)) - as malloc, но инициализирует массив с нулями
- **realloc**(ptr, size) - меняет выделенный размер, новая область не инициализируется

### Pointers

Указатели используются для указания на динамически выделеную память во время выполнения.

Объявленные указатели, как и обычные локальные переменные выделяются в автоматической стэковой области памяти.
Когда вызывается функция для её локальных переменных выделяется память в Stack-е, локальные переменные освобождаются при завершении выполнения функции.

Pointer-ы указываются так:
```c
type * name = address;
```

`*` - is called "asterics", but in A* algorithm it's called "star"
`&` - "ampersand"

- ptr - address of value to which pointer is pointing
- &ptr - address of pointer itself

- *ptr - dereferencing the pointing value

---

### Pointer to pointer

Информация об указателе.
Реализуется с помощью указателя на указатель.

Указатель на указатель ptr хранит адрес.

- ptr (&*ptr) - адрес оригинального указателя
- &ptr - адрес нового созданного указателя на указатель

- *ptr - оригинальный указатель, при доступе вернет адрес variable на который указывает этот оригинальный указатель
- **ptr - dereferencing, basically

See an example in `pointer.c` in function `print_dublicate_pointer_info`.


### Висячий указатель (Dangling Pointer)

> Do not use freed pointer.

Проблема относится именно к динамическому выделению памяти.

Использования указателя после освобождения памяти на который он указывает.

После освобождения выделенной памяти указатель всё ещё указывает на область памяти, которая уже может быть занята не им.

```c
int * ptr = malloc(8);

// ptr = 0x01234ABC
// &ptr = 0x56789DEF

free(ptr);

// ptr = 0x01234ABC, still
// &ptr = 0x56789DEF
```

Поэтому иногда reference value of freed pointer присваивают NULL:
```c
free(ptr);
ptr = NULL;
```

See an example in `free.c`.


---


