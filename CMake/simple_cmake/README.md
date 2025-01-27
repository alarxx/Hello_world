# CMake

---
<- [[Make (Software)]]

"Cross-platform Make"

---

```bash
sudo apt install cmake
cmake --version
```

---

**CMake генерирует [[Make (Software) - Makefile|Makefile]] или [[Ninja (C++)]].**

CMake creates Makefile,
после чего мы можем пользоваться Make для сборки проекта,
пока не изменим структуру проекта, например добавим source или header файлы или зависимости.

**Out-of-source builds**
Makefile-ы генерируют не в src/, а в build/ директорию, чтобы не перегружать root директорию проекта.

---

**CMakeLists.txt** - configuration file

![[Gary Explains - CMake Tutorial for Absolute Beginners - From GCC to CMake including Make and Ninja 14m52s.png|500]]

CMakeLists.txt:
```c
cmake_minimum_required(VERSION 3.10)

project(MY_PROJECT)

add_executable(${PROJECT_NAME} main.cpp)
```

---

**Usage:**

```sh
cmake [options] -S <path-to-source> -B <path-to-build>
```

```sh
cd src

mkdir build
cd ./build

cmake ..
or
cmake -S ../ -B ./

make
./main
```

Ещё CMake может генерировать и для [[Ninja (C++)]]:
```sh
mkdir ninja
cd ../ninja
cmake -G Ninja ..
ninja
./main
```

Ninja born from work on the [[Chromium Browser]].

Ment to replace [[Make (Software)]], x10 times faster.

```sh
sudo apt install ninja-build
ninja --version
```

---

See also:
- [[Gary Explains (Youtube) - CMake Tutorial for Absolute Beginners]]
- Playlist: [[Code, Tech, and Tutorials (Youtube) - CMAKE TUTORIAL]]
-
- [[Make (Software)]]
- [[Make (Software) - Makefile]]
-
- [[Software Packaging, Build and Deployment System]]
- [[Continuous Integration (CI)]]
-
- Book: [[Martin (Book) - Mastering CMake]]

- Build [[OpenCV]]
