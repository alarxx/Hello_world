# My Math Library

_libmymath.a_ - Static Library File (Archive).

Library include name is `mymath`.


С названиями немного запутано.
Название проекта библиотеки - mymathlib, оно не на что здесь не влияет.
Linkовщику мы говорим название mymath.
Дальше CMake добавляет сначала lib и build-ит archive файл с названием libmymath.a,
но link-уем с приложением и включаем в код по названию, которое дали линковщику.

## Building Library

Build:
```c
mkdir build
cd build
cmake ..
make
```

Получается, у нас сгенерируется статичная библиотека libmymath.a:
1. building object files (.o)  
2. linking them to static library (.a)  
Заметь, что мы написали название архива `mymath`, а получили `lib`+`mymath.a`.

Как включить статичную библиотеку в исполняемый файл программы?
```cpp
#include "../mymath/adder.h"
```

Этот подход подразумевает, что мы укажем путь к archive файлу библиотеки при её использовании в приложении.

Другой подход - мы можем установить библиотеку в стандартном расположении.


## Installation in standard location

```sh
su -c "make install" root
```
После этого библиотека установится в local standard location:
- /usr/local/lib/libmymath.
- /usr/local/include/mymath/adder.h


Так как библиотека в стандартном расположении, её можно включать в код так:
```cpp
#include <mymath/adder.h>
```
Instead of providing relative path,
compiler will automatically find the library.

