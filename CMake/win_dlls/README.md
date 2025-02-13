# Developing on Windows

Неудобно!

1. Dowload Cygwin
2. Download dependencies: git, openssh, gcc, g++, MinGW...gcc, cmake, make
3. Set up git ssh
4. Compile and run using GCC

Почему-то добавляет к библиотекам prefix cyg, а не lib, типа чтобы не перепутать:
- cygSqrtLibrary.dll
- cygMathFunctions.dll

Оказывается, .exe по умолчанию ищут .dll относительно своей папки.

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