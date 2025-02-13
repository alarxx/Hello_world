# Developing on Windows

Неудобно!

1. Dowload Cygwin
2. Download dependencies: git, openssh, gcc, g++, MinGW...gcc, cmake, make
3. Set up git ssh
4. Compile and run using GCC

Почему-то при сборке с CMake добавляет к библиотекам prefix `cyg*`, а не `lib*`, типа чтобы не перепутать:
- cygSqrtLibrary.dll  
- cygMathFunctions.dll  
Я не разобрался, как сделать нормально.

Еще есть MSYS2, но он тоже добавляет префикс - `msys2_*`, вроде.

У меня не получилось настроить RPATH так, как мы делали, .dll не находилась.  
Оказывается, .exe по умолчанию ищут .dll относительно своей папки.

---

Ещё есть вот такие штуки для функций:  
- `__declspec(dllexport)`  
- `__declspec(dllimport)`  
И без _WIN32  
- `__attribute__ ((visibility ("default")))`  
Но пока все без них работало...

---

```sh
#! /bin/sh

cd mymathlib
# Position Independent Code (PIC), но и без этого работает почему-то
g++ mymath.cpp -c -fPIC
g++ mymath.o -o libmymath.dll -shared

echo Shared library generated!

cd ../
g++ main.cpp -c
g++ main.o -o app -lmymath -L./mymathlib 
# -Wl,-rpath=. - does not work, idk
# ld -rpath=.

echo "app's shared object dependencies:"

mv ./mymathlib/libmymath.dll ./

# ldd - print shared object dependencies, idk
ldd app
```

