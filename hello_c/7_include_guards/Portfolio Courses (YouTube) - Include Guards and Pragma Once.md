---
creation_time: 2024-07-29 17:04
bib_type: "[[Youtube Video]]"
bib_author:
  - Portfolio Courses
bib_date: 2023-05-16
bib_title: " Include Guards | C Programming Tutorial"
bib_link: https://www.youtube.com/watch?v=pF1_fXz5zN0
topics:
  - "[[C Programming Language]]"
  - "[[C Programming Language - Modules]]"
---
Prev: [[Portfolio Courses (YouTube) - How To Create a Library and Split a Program Across Multiple Files]].


---

# Preprocessor Include Guards

| student.h :
```c
#ifndef STUDENT_H
#define STUDENT_H
typedef struct {
    char name[100];
    int id;
} Student;
#endif

```

| course.h :
```c
#ifndef COURSE_H
    #define COURSE_H

        #include "student.h"

        typedef struct {
            Student students[100];
            char name[100];
        } Course;

#endif

```


| main.c :
```c
#include <stdio.h>

#include "course.h"
#include "student.h"

int main(void){
    return 0;
}

// gcc -Wall main.c -o program
```

>If by chance somewhere else in our code a macro called "COURSE_H" was defined this could break, so we really need to be careful that the name we use are unique and only used for this purpose.

Нужно следить за именами макросов.

---
# Pragma Once

pragmatic

>There is another way of guarding include files. In C there is a pre-processor directive called "pragma", that allows us to give additional information to the compiler. Not all pragma directive are supported by all compilers but a very widely supported pragma directive is the pragma once directive.

```c
#pragma once
#include "student.h"

typedef struct {
Student students[100];
char name[100];
} Course;
```

> `#pragma once` is going to have exact same result as our include guards, it is going to ensure that the contents of the header files included only once. It is universal even though it's not officially part of the C standard.

According to [/stackoverflow](https://stackoverflow.com/questions/5776910/what-does-pragma-once-mean-in-c) we can add both:
```c
#pragma once
#ifndef _MYHEADER_H_
#define _MYHEADER_H_
...
#endif
```

P.S.: ligma

