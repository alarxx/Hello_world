---
creation_time: 2024-07-29 15:49
bib_type: "[[Youtube Video]]"
bib_author:
  - Portfolio Courses
bib_date: 2022-01-05
bib_title: How To Create A Library And Split A Program Across Multiple Files | C Programming Tutorial
bib_link: https://www.youtube.com/watch?v=x8gsHFBW7zY
topics:
  - "[[C Programming Language]]"
  - "[[C Programming Language - Modules]]"
---

In programming as we write programs that becomes larger and larger it is very important to split up programs across multiple files. These files are called differently in different programming languages and technologies that you use, for example: components, modules or libraries. 

В C такие файлы называются библиотеками (libraries).

---
От меня.

В Java стандартно почти каждый файл репрезентирует отдельный класс или интерфейс. Такие файлы объединяются в пакеты, а пакеты в модули. 

ChatGPT:

**Class**: MyClass.java
```java
package com.example.myapp;

public class MyClass {
    // class implementation
}
```

**Package**: `com.example.myapp`
Contains MyClass.java, AnotherClass.java, etc.

**Module**: `module-info.java`
```java
module com.example.myapp {
	exports com.example.myapp;
}
```

---

| library.c :
```c
#include <stdio.h>

int add(int a, int b){
    printf("addition function\n");
    return a + b;
}
```

| library.h
```c
int add(int a, int b);
```

| main.c :
```c
#include <stdio.h>
#include "library.h"

int main(){
    int a = 2, b = 3;
    int c = add(a, b);
    printf("c = %i \n", c);
}
```

| Compile:
```sh
gcc -Wall main.c library.c -o main.bin
```
| Execute:
```sh
./main.bin
```

---

Next:
- [[Portfolio Courses (YouTube) - Include Guards]]
