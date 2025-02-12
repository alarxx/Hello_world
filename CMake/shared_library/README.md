# Shared Library

https://www3.ntu.edu.sg/home/ehchua/programming/cpp/gcc_make.html

- **Static Libraries:**
	- .a (archive file) - linux
	- .lib (library) - windows
- **Shared Libraries:**
	- .so (shared objects) - linux
	- .dll (dynamic link library) - windows
- **Header-only library** (static)

Advantages of shared libraries:
1. Disk space preservation
2. RAM preservation
3. Ease of maintenance

**Searching headers and libraries**
- `-Idir`: (_include-paths_) где искать header-files of `#include`
- `-Ldir`: (_library-paths_) где искать библиотеки
- `-lxxx` and `-lxxx.lib`: for libraries itself

---


## Commands for this repo

**Как компилировать статические и динамические библиотеки**

- https://www.youtube.com/watch?v=Slfwk28vhws
- https://stackoverflow.com/questions/10358745/how-to-use-libraries

---

**Static libraries** are created by simply archiving the `*.o` files with the `ar` program:
```sh
# Create the object files (only one here)
g++ -c mymath.cpp
# Create the archive (insert the lib prefix)
ar rcs libmymath.a mymath.o
```

---

**Shared libraries** are created with the `g++`, `-fpic`, and `-shared` options:

```sh
g++ main.cpp -c

# nm - list symbols from object files
nm main.o
# На C красиво выходит, на C++ name mangling
```

Compile Shared Library:
```sh
# Create the object file with Position Independent Code[**]
g++ mymath.cpp -c -fpic
# Crate the shared library (insert the lib prefix)
g++ mymath.o -o libmymath.so -shared

# file - determine file type
file libmymath.so
# libmymath.so: ELF 64-bit LSB shared object, x86-64
```

Link with library:
```sh
g++ main.cpp -c
g++ main.o -o app -lmymath -L. -Wl,-rpath=.
# -Wl,-rpath=. - insert the path to the shared library into the binary
# ld -rpath=.

# nm - list symbols from object files
nm -D app

# ldd - print shared object dependencies
ldd app
# libmymath.so => not found or <dir>
```

```sh
# если не использовали -rpath при линковке, то нужно указать путь к shared libraries,
LD_LIBRARY_PATH=./ ./app
```
