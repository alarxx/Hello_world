---
creation_time: 2024-06-13 19:31
parents:
  - "[[Programming Language]]"
---
---

Outline (Agenda)
- [[C++ Programming Language#Main Steps|Main Steps]].
- [[C++ Programming Language#Compilation|Compilation]].
- [[C++ Programming Language#Hello World|Hello World]].
- [[C++ Programming Language#Pointers]].
- [[C++ Programming Language#Preprocessor|Preprocessor]].
- [[C++ Programming Language#Features|Features]].
	- [[C++ Programming Language#inline-функции|inline-функции]].
	- [[C++ Programming Language#template|template]].
	- [[C++ Programming Language#const|const]].
	- [[C++ Programming Language#typedef - type alias|typedef - type alias]].
	- [[C++ Programming Language#struct|struct]].

---
# Main Steps

- Preprocessing (.h files)
- Compilation
- Linking to binary execution file
- Execution

---

# Compilation

Для компиляции C++ можно использовать [[GNU Compiler Collection (GCC)|GCC]].

```sh
gcc main.cpp -o main -lstdc++
```
- -lstdc++ - link lib std c++ .so

Либо просто использовать g++ компилятор, который по умолчанию использует C++:
```sh
g++ main.cpp -o main
```

**Parameters**
```sh
g++ -std=c++17 -Wall -pedantic main.cpp utils.cpp -o program
```
- -std - version to use
- -pedantic - строгая проверка соответствия стандарту
- -Wall - предупреждения компилятора (анализ кода)

Debugging [[GNU Debugger (GDB)|gdb]]:
```sh
gcc -g main.cpp -o program
```
- -g 
```
gdb ./program
```

Optimization:
```sh
g++ -O2 main.cpp -o program
```
- -O2 - [/Options That Control Optimization](https://stackoverflow.com/questions/655382/meaning-of-gcc-o2)

---

# Hello World

main.cpp:
```c++
#include <iostream>

int main(int argc, char* argv[]) {
	std::cout << "Hello, World!" << std::endl;

    std::cout << "Argument Count: " << argc << "\n";
    for (int i = 0; i < argc; ++i) {
        std::cout << "Argument[" << i << "]: " << argv[i] << "\n";
    }

    return 0;
}

```

Compile:
```sh
g++ main.cpp -o program
```

```sh
./program arg1 arg2 arg3
```

---

# Pointers

```c

1. Статическое выделение памяти
С помощью keyword static память выделяется статически.
Статически выделенная память выделяется во время компиляции/запуска программы и не может быть изменена.

2. Автоматическое выделение памяти - In Stack
Для локальных переменных функции выделяется память в стеке, которая после удаляется при завершении функции.

3. Динамическое выделение памяти - In Heap

- **malloc** (n * sizeof(type)) - allocates in bytes with garbage values
- **free**(pointer to allocated memory)
- **calloc**(n, sizeof(type)) - as malloc, но инициализирует массив с нулями
- **realloc**(ptr, size) - меняет выделенный размер, новая область не инициализируется

Указатели используются для указания на динамически выделеную память во время выполнения.
Объявленные указатели, как и обычные локальные переменные выделяются в автоматической стэковой области памяти.

Когда вызывается функция для её локальных переменных выделяется память в Stack-е, локальные переменные освобождаются при завершении выполнения функции.
```

```c
int myvar = 5;
int * ptr = &myvar;

ptr - address of myvar
&ptr - address of pointer itself
*ptr - dereferencing - value of myvar

Функция создает новый поинтер
void fun(int * arg_ptr){
	...
}
fun(&ptr)
argptr = &ptr
&argptr != &ptr


struct MyStruct {
	char* (*fun)(*void); 
};
char * fun(int * num){}
struct MyStruct obj = {fun};
```

---

# Preprocessor

**Macros**

(до появления `inline`) это был способ сократить повторение кода и избежать вызовов функций.
```c++
#define SQUARE(x) ((x) * (x))
```

```c++
std::cout << SQUARE(5.5) << std::endl; // Работает для double 
std::cout << SQUARE(4) << std::endl; // Работает для int
```

Проблемы
```c++
std::cout << SQUARE("Hello") << std::endl;  // Ошибка компиляции

int i = 2;
// (++i * ++i) = 16 (увеличение i дважды)
std::cout << SQUARE(++i) << std::endl;  
```

Можно заменить использованием inline-функций или шаблонами.

Иногда использование оправдано:
1.
```c++
#ifdef _WIN32
#define PLATFORM "Windows"
#else
#define PLATFORM "Unix-based"
#endif
```
2.
```c++
#define DEBUG_MSG(x) std::cout << "[DEBUG] " << x << std::endl;

int main() {
    DEBUG_MSG("Program started");
}
```

---

**Include Guards**

```c
Почему-то g++ компилирует без ошибок без Include Guards

Include Guards
/#ifndef

Pragmatic
/#pragma once
Еще легче использовать, это облегчает naming-и
Но, не знаю как будет работать, если файлы будут одинакого называться

#ifndef UTILS_H
#define UTILS_H

#include <string>

#define SQUARE(x) ((x) * (x))

// Прототип функции
std::string greet(const std::string& name);

#endif
```

---

# Features

## inline-функции

Рекомендация компилятору попытаться встроить тело функции, чтобы избежать накладных расходов (передача функции в Stack).
```c++
inline int square(int x) { 
	return x * x; 
} 
int main() { 
	for(int i=0; i<100; i++){
		square(5); 
	}
}
```
Some compilers will automatically replace function call by function body

Почему-то функции класса всегда считаются inline.

```c++
class Math { 
public: 
	int square(int x) { 
		return x * x; // inline по умолчанию 
	} 
};
```

----

## template

```c++
template <typename T> 
T square(T x) { 
	return x * x; 
} 

int main() { 
	std::cout << square(5) << std::endl; // int 
	std::cout << square(5.5) << std::endl; // double 
}
```

---

## const

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

---

## typedef - type alias

```c
typedef unsigned int uint
uint x = 10; 

#define PI 3.14159 
#define SQUARE(x) ((x) * (x))
```

---

## struct

```c
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
    
    free(shape);
	shape = NULL; // Prevent Dangling Pointer

    return 0;
}


```


---

## Literal Suffixes

u
l - long int
ll
ul
ull

f
d
l - long double

Complex numbers
i - float 123.0i
il - long double
if - float?

0 - octal(8) 0123
0x - hexadecimal(16) 0x123
0b - binary(2) 0b1011

# 

---
# Interesting

.**ENV**
```c++
#include <fstream> 
#include <iostream> 
#include <cstdlib> 
#include <string>

void load_env(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Unable to open .env file\n";
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        auto delimiter_pos = line.find('=');
        if (delimiter_pos != std::string::npos) {
            std::string key = line.substr(0, delimiter_pos);
            std::string value = line.substr(delimiter_pos + 1);

            // Устанавливаем переменную окружения
			setenv(key.c_str(), value.c_str(), 1); 
        }
    }
}

int main() {
    load_env(".env");

    const char* value = std::getenv("MY_VAR");
    if (value) {
        std::cout << "MY_VAR: " << value << std::endl;
    } else {
        std::cout << "MY_VAR is not set" << std::endl;
    }

    return 0;
}
```