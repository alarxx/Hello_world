# Include guards

> Double inclusion problem

Поверхностно затрагиваем тему препроцессора.

Without include guards:
```
In file included from main.c:4:
student.h:5:3: error: conflicting types for ‘Student’; have ‘struct <anonymous>’
    5 | } Student;
      |   ^~~~~~~
In file included from course.h:1,
                 from main.c:3:
student.h:5:3: note: previous declaration of ‘Student’ with type ‘Student’
    5 | } Student;
      |   ^~~~~~~
```

References:

1. [[Portfolio Courses (YouTube) - How To Create a Library and Split a Program Across Multiple Files.md]]

2. [[Portfolio Courses (YouTube) - Include Guards.md]]

3. [/Geekforgeeks](https://www.geeksforgeeks.org/cc-preprocessors/);

4. [/Metanit](https://metanit.com/c/tutorial/3.1.php);

