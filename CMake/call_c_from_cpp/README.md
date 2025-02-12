# extern "C"

https://stackoverflow.com/questions/13694605/how-to-use-c-source-files-in-a-c-project/51912672#51912672

C++ использует name mangling (`add` -> `_Z3addii`), а C нет.
Если мы захотим использовать библиотеку написанную на C из C++,
то нужно учитывать это.

## Call C from C++

Представим, что мы пишем на C++, и мы хотим вызвать функцию из библиотеки C, её название `sub`.
Если мы подключим нужный header file в нашем коде, и он декларирует функцию, но её название изменится,
из-за чего мы не сможем правильно слинковать программу.

`extern "C"` используется, чтобы убрать name mangling.

Первый подход это обернуть включение:
```cpp
extern "C" {
#   include "legacy_C_header.h"
}
```

Второй подход, когда сам header подразумевает использование из C++:
```c
#pragma once
/* ligma */

#ifdef __cplusplus
extern "C" {
#endif

/* all of your legacy C code here */

#ifdef __cplusplus
}
#endif
```

---

## Call C++ from C

Если мы хотим использовать .hpp из C:
```cpp
#pragma once

#ifdef __cplusplus
// C can't see these overloaded prototypes
int f(int a);
int f(float a);

extern "C" {
#endif

f_int(int i);
f_float(float i)

#ifdef __cplusplus
}
#endif
```

То есть мы прячем то, что запрещено в C,
и mangling-ируем имена функций подразумевающихся для C.
