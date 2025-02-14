# Official CMake Tutorial

release (3.31.5)

https://cmake.org/cmake/help/latest/guide/tutorial/index.html


----


# Commands

- cmake_minimum_required
- project
- set
- configure_file(in.h -> .h)
- add_executable

**How to link library?**
- add_library

- add_subdirectory
- `link(APPEND <var_name> <value>)`
- `target_link_libraries(<target> <visibility> <dependency-list>)`
- `target_include_directories(<target> <visibility> <header-dir-list>)`

- option
- if(), endif()
- target_compile_definitions - pass to target, to be accessible from source code
- target_sources, or use like static library

- find_library vs. target_link_directories?
- find_package?

**Visibility:**
- PUBLIC - those who depend on the executable will have access
- PRIVATE - those who depend on the executable won't have access
- INTERFACE - library interface, we don't include by default, we can specify a relative path, but those who depend will have default access

**Library types:**
- STATIC - compiles on linking stage
- SHARED - compiles on loading the library
- INTERFACE - never compiles

- target_compile_features

etc...

----

# Step 1

- add_executable
- c++ standard
- versioning

_`AppConfig.h.in`_: (input)
```c
#define App_VERSION_MAJOR @App_VERSION_MAJOR@
#define App_VERSION_MINOR @App_VERSION_MINOR@
#define MY_CUSTOM_VARIABLE "@MY_CUSTOM_VARIABLE@"
```

_`main.cpp`_:
```cpp
#include <string>
#include <AppConfig.h>
int main(){
	std::string str = MY_CUSTOM_VARIABLE; // "123" - string
	// C++11 feature
	int num = std::stoi(str); // 123 - int
}
```

Build and Run:
```sh
mkdir build
cd build
cmake ..
make
./app
```

---

# Step 2. Adding a Library

- add_library
- option

#### CMakeLists

_`CMakeLists.txt`_:
```c

```


_`MathFunctions/CMakeLists.txt`_:
```c

```

Option можно включать или отключать из [[CMake GUI]] или командной строки с flag-ом `-D`.

---

#### Usage

Теперь в зависимости от option переменной можно менять реализацию функции.

**_`MathFunctions/MathFunctions.cxx`_:**
```cpp
#include "MathFunctions.h"

#include <cmath>

#ifdef USE_MYMATH
#  include "mysqrt.h"
#endif

void print_cxx_std();

namespace mathfunctions {
  double sqrt(double x)
  {
	print_cxx_std();
	// Это будет выполняться на стадии препроцессинга, кажется
	// Получается, в pure C++ из них попадет только одна инструкция
	#ifdef USE_MYMATH
		return detail::mysqrt(x);
	#else
		return std::sqrt(x);
	#endif
	}
}

```


**_`tutorial.cxx`_:**
```cpp
#include <cmath>
// #include <cstdlib> // atof
#include <iostream>
#include <string>

#include "TutorialConfig.h"
#include "MathFunctions.h"

using std::cout, std::endl;

void print_cxx_std();

int main(int argc, char* argv[]){
  print_cxx_std();

  cout << "Version: " << Tutorial_VERSION_MAJOR << "." << Tutorial_VERSION_MINOR << endl;
  cout << MY_CUSTOM_VARIABLE << endl;

  #ifdef USE_MYMATH
    cout << "USE_MYMATH" << endl;
  #else
    cout << "USE_MYMATH not defined" << endl;
  #endif

  // convert string to double
  // const double inputValue = atof(argv[1]);
  const double inputValue = std::stod("25"); // C++11 feature

  // calculate square root
  // const double outputValue = sqrt(inputValue);
  const double outputValue = mathfunctions::sqrt(inputValue);
  cout << "The square root of " << inputValue << " is " << outputValue << endl;

  return 0;
}
```

---

Print C++ Standard:
```cpp
void print_cxx_std(){
// https://stackoverflow.com/questions/2324658/how-to-determine-the-version-of-the-c-standard-used-by-the-compiler
  if (__cplusplus == 202101L) std::cout << "C++23";
  else if (__cplusplus == 202002L) std::cout << "C++20";
  else if (__cplusplus == 201703L) std::cout << "C++17";
  else if (__cplusplus == 201402L) std::cout << "C++14";
  else if (__cplusplus == 201103L) std::cout << "C++11";
  else if (__cplusplus == 199711L) std::cout << "C++98";
  else std::cout << "pre-standard C++." << __cplusplus;
  std::cout << "\n";
}
```

---

# Step 3. Target Usage Requirements

**INTERFACE**

```c
target_include_directories(<libary> INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
```
Прикольная штука, получается можно использовать INTERFACE (когда target не видет, но зависимости видят), как requirement для включения.

Для включения библиотеки хватает только:
```c
target_link_libraries(<target> <visibility> <library>)
```

---

**"Requirements"**

**`INTERFACE` library** не имеет кода и не комплириуется в объектный файл, она нужна только для распространения своих свойств.

Setting C++11 Standard (98, 11, 14, 17, 20, 23, 26)
```cmake
add_library(tutorial_compiler_flags INTERFACE)
target_compile_features(tutorial_compiler_flags INTERFACE cxx_std_11)
...
target_link_libraries(<target> <visibility> ${libs} tutorial_compiler_flags)
```
Но, это кажется не required, у меня остается C++17 - версия стандарта по умолчанию.

Интересно, что эта библиотека доступна из nested CMakeLists.txt.

---

- STATIC - compiles on linking stage
- SHARED - compiles on loading the library
- INTERFACE - never compiles

----

# Step 4. Generator Expressions

**Minimum version**
```c
cmake_minimum_required(VERSION 3.15)
```

**Generator:**
`"$<0:...>"`- results in an empty string
`"$<1:...>"` - resuls in the content of `...`

**Using:**
- [`$<COMPILE_LANG_AND_ID:language,compiler_ids>`](https://cmake.org/cmake/help/latest/manual/cmake-generator-expressions.7.html#genex:COMPILE_LANG_AND_ID):
- BUILD_INTERFACE

Flags:
```c
set(gcc_like_cxx "$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang,ARMClang,LCC>")
# Microsoft Visual C++, другие флаги
set(msvc_cxx "$<COMPILE_LANG_AND_ID:CXX,MSVC>")
```
Кажется, должно быть точное совпадение, поэтому можно сначала написать [[Clang]], а потом уже AppleClang и так далее.

**Add Warning flags to `INTERFACE library`** when building, but not for installed versions (`BUILD_INTERFACE`):
```c
target_compile_options(tutorial_compiler_flags INTERFACE
    "$<${gcc_like_cxx}:$<BUILD_INTERFACE:-Wall;-Wextra;-Wshadow;-Wformat=2;-Wunused>>"
    "$<${msvc_cxx}:$<BUILD_INTERFACE:-W3>>"
)
```
- `"a,b"` - обычный символ, но в генераторах - разделители аргументов
- `"a;b"` - отдельные строки "a" и "b" для списка

Although, вместо генераторов можно использовать обычные if() endif():
```cpp
# Check compiler: GNU, Clang, AppleClang, Intel, MSVC
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # using clang with clang-cl ("MVSC") or clang native ("GNU")?
endif()
```
https://stackoverflow.com/questions/10046114/in-cmake-how-can-i-test-if-the-compiler-is-clang

---

# Step 5. Install and Test

## Installing

Building:
```sh
mkdir build
cd build
cmake -S .. -B .
cmake --build .
cmake --install . --config Release
# works only with root
# su -c "make install" root
```
Full all-in-one, as IDE's `INSTALL` would do:
```sh
cmake --build . --target install --config Debug
```

Use `--prefix` to override install location
```sh
cmake --install . --prefix "/home/<...>/installdir" --config Debug
```

Output:
```
[ 33%] Built target SqrtLibrary
[ 66%] Built target MathFunctions
[100%] Built target Tutorial
Install the project...
-- Install configuration: ""
-- Installing: /usr/local/libs/libMathFunctions.a
-- Installing: /usr/local/libs/libSqrtLibrary.a
-- Installing: /usr/local/include/MathFunctions.h
-- Installing: /usr/local/bin/Tutorial
-- Installing: /usr/local/include/TutorialConfig.h
```

---

**Может ли быть несколько `install`?**

Да, с помощью
- `install(... COMPONENT <component>)`
- `cmake ... --component <component>`

-> CPack

---

[[Static and Dynamic Linking]]
- `ARCHIVE DESTINATION` - static: `.a`, `.lib`
- `LIBRARY DESTINATION` - dynamic: `.so`, `.dll`
- `RUNTIME DESTINATION` - windows `.dll`? `.exe`, `.deb`, `.dmg`, [[Red Hat Package Manager (RPM)|.rpm]]

---

```c
message("PROJECT_BINARY_DIR: ${PROJECT_BINARY_DIR}")
message("PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")
message("CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")
message("CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
message("CMAKE_CURRENT_BINARY_DIR: ${CMAKE_CURRENT_BINARY_DIR}")
```

Output:
```
PROJECT_BINARY_DIR: /home/<...>/Step2/build
PROJECT_SOURCE_DIR: /home/<...>/Step2
CMAKE_SOURCE_DIR: /home/<...>/Step2
CMAKE_CURRENT_SOURCE_DIR: /home/<...>/Step2/MathFunctions
CMAKE_CURRENT_BINARY_DIR: /home/<...>/Step2/build/MathFunctions
```

---

## Testing
[[Software Quality Assurance and Testing (SQaT)]]

Manage unit tests, they can be added with command:
- `add_test()`

There is a lot of compatibility between CTest and Google Test.
- [[CTest]]
- [[Google Test]]

Commands:
- `enable_testing`
- `add_test`
- `function`
- `set_tests_properties`
- `ctest`

Steps:
- enable_testing
- add_test
- rebuild `cmake ..`
- `ctest -N|--show-only`
- `ctest -VV|--extra-verbose`
- specify configuration `-C|--build-config <config>`: Debug, Release
Or use `RUN_TESTS` from the IDE

---

[[Regular Expression (RegEx)]]

---

# Step 6. Display test results Dashboard

Run tests and submit results to [[CDash]]

add `include(CTest)`
remove `enable_testing()`

_`CTestConfig.cmake`_?
```python
set(CTEST_PROJECT_NAME "CMakeTutorial")
# The time when a 24 hour "day" starts for this project
set(CTEST_NIGHTLY_START_TIME "00:00:00 EST")

# URL of the CDash
set(CTEST_DROP_METHOD "http")
set(CTEST_DROP_SITE "my.cdash.org")
set(CTEST_DROP_LOCATION "/submit.php?project=CMakeTutorial")
set(CTEST_DROP_SITE_CDASH TRUE)
```

Run:
```sh
ctest [-VV] [-C Debug] -D Experimental
```

`-D <category>`

Check on Kitware's public dashboard
https://my.cdash.org/index.php?project=CMakeTutorial

---

# Step 7. System Introspection

С помощью модуля CheckCXXSourceCompiles и его функции check_cxx_source_compiles можно проверить ==скомпилируется ли== определенный блок кода:
```cpp
include(CheckCXXSourceCompiles)

check_cxx_source_compiles("
	#include <cmath>
	int main() {
		std::log(1.0);
		return 0;
	}
" HAVE_LOG)

...

if(HAVE_LOG AND HAVE_EXP)
	target_compile_definitions(SqrtLibrary
		PRIVATE "HAVE_LOG"
	)
endif()
```
HAVE_LOG теперь хранит результат выполнения (0|1) и передает это значение в source code.

Дальше мы мы можем использовать это из кода
```cpp
#if defined(HAVE_LOG) && defined(HAVE_EXP)
	...
#elif defined(x)
	...
#else
	...
#endif
```

Это может быть полезно, если нужно что-то проверить во время компиляции и в зависимости от этого менять реализацию кода. Функции зависимые от платформы, например и так далее.
Например, если я использую CUDA, то такая реализация кода, а если я использую OpenCL или вообще тупо обычный CPU, то другая реализация кода.

---

Оказывается, команда `target_compile_definitions` просто `#define` делает, чтобы конкретное значение установить нужно делать так:
```c
target_compile_definitions(<target> <visibility> <variable>=<value>)
```

----

# Step 8

Основная идея тут сгенерировать header файл с массивом в котором хранятся заранее посчитанные квадратные корни.

**Modules**
Оказывается, можно создавать "модули" в CMake:
```c
include(MakeTable.cmake)
```
И потом MakeTable.cmake расскроется просто как header files.

**Custom Command**
Дальше они добавляют новую библиотеку, которая вызывает саму себя для генерации файла:
```c
add_executable(MakeTable MakeTable.cxx)

# Выполняется во время компиляции, а не генерации Makefile
add_custom_command(
	OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/Table.h
	COMMAND MakeTable ${CMAKE_CURRENT_BINARY_DIR}/Table.h
	DEPENDS MakeTable
)
# Output тут видимо нужен, чтобы не перекомпилировать одно и то же, как target в Make
```

**Generating files**
Hardcode-ят строки в файл.
Типа
```cpp
fout << sqrt(static_cast<double>(i)) << "," << std::endl;
```

---

# Step 9. CPack

`cmake --install .` устанавливает программу локально, в то время как
**CPack создает установочный файл**, чтобы другие могли скачать программу.

CMake команды `install()` указывают какие файлы куда установить, а cpack упаковывает, то есть либо создает архив, либо создает установщик.

---

```c
include(InstallRequiredSystemLibraries)
# IDK why licence is only visible in the archives
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_RPM_PACKAGE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_PACKAGE_VERSION_MAJOR "${Tutorial_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${Tutorial_VERSION_MINOR}")
set(CPACK_GENERATOR "ZIP;TGZ;DEB;RPM;NSIS;IFW")
set(CPACK_SOURCE_GENERATOR "ZIP;TGZ;DEB;RPM;NSIS;IFW")

# CPACK_PACKAGE_CONTACT
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Alar Akilbekov alar.akilbekov@gmail.com")

set(CPACK_IFW_VERBOSE ON)
# Do I need to set version as a string?
set(CPACK_IFW_FRAMEWORK_VERSION 4.8.1)
set(CPACK_IFW_ROOT "/Qt/QtIFW-4.8.1/")

include(CPack)
```

---

ChatGPT: `CPACK_RESOURCE_FILE_LICENSE` только показывает лицензию в GUI-инсталляторах (NSIS, IFW), но **не добавляет файл в архив**.

Лицензионного соглашения нет в упакованном архиве, чтобы добавить:
```c
install(
	FILES "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE"
	DESTINATION share/doc/<project>
)
```

---

**Specify configuration**
```sh
cpack -C Debug
```

---

**Generators**
```sh
cpack --help
```

- ZIP
- TGZ = tar.gz
-
- DEB - [[Advanced Package Tool (APT)|apt install]]
- [[Red Hat Package Manager (RPM)|RPM]]
- NSIS / NSIS64 = Null Software Installer for win.exe
- IFW = Qt Installer Framework for cross-platform installer with GUI

Подожди, а архив это что вообще, там просто хранятся типа бинарники и интерфейсы? А NSIS и IFW это что? А DEB это что? Это установщик же? Да.

**Archives** - это просто упакованные файлы, которые ты указываешь с `install()`.
```sh
cpack -G ZIP
```

Чем отличаются разные архивы? tar, zip, rar
- TAR - Unix
- ZIP - Windows
- RAR - Windows

**Application Installer**

**DEB**
https://cmake.org/cmake/help/latest/cpack_gen/deb.html

Обязательно нужно указать maintainer-а:
```c
# CPACK_PACKAGE_CONTACT
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Alar Akilbekov alar.akilbekov@gmail.com")
```

Как установить .deb:
```sh
apt install ./<pkg>.deb
```

**[[Red Hat Package Manager (RPM)]]**

Я с Debian, и мне нужно было предварительно установить RPM:
```sh
apt install rpm
```

**[[Qt Installer Framework]]**
Вроде работает, но нужно еще покопать.

----

# Step 10. Shared Libraries

Если не указать, по-умолчанию же все библиотеки создаются STATIC, можно это изменить:
```c
option(BUILD_SHARED_LIBS "Build using shared libraries" ON)
```

Попробуй забилдить, все должно работать. Прямо в Tutorial должен записаться путь к библиотекам.

Но, попробуй установить и тебе придется указывать путь к .so библиотекам, либо можно их закинуть flat в стандартный путь /lib/, тогда будет тоже работать.
Мне не нравится закидывать flat в /lib/. Мне вообще не особо нравится использовать shared libraries, столько мароки, и ради чего?

---

Дохрена нюансов shared библиотек для windows.
`_WIN32`?
`__declspec(dllexport)` перед функцией, чтобы экспортировать её ?

==Оказывается еще есть Runtime Linking==
[[C C++ Programming Language - extern C and  name mangling]]

[[Runtime Linking]]

---

# Step 11. Adding Export Configuration

Настраиваем проект так, чтобы другие CMake проекты могли импортировать его используя `find_package()`.

**`find_library()` vs. `find_package()`**
`find_package()` не просто находит библиотеку, но и настраивает header файлы, зависимости и т.д.

К `install()` нужно добавить `EXPORT`, который сгенерирует ==**_`MathFunctionsTargets.cmake`_**== с кодом для импорта перечисленных target-ов:
**_`MathFunctions/CMakeLists.txt`_** :
```c
install(
	TARGETS ${installable_libs}
	EXPORT MathFunctionsTargets
	DESTINATION lib
)
```
И после нужно установить этот файл:
**_`CMakeLists.txt`_** : (но думаю можно и не из root директории???)
```c
install(
	EXPORT MathFunctionsTargets
	FILE MathFunctionsTargets.cmake
	DESTINATION lib/cmake/MathFunctions
)
```

И когда мы подключаем нашу библиотеку через `find_package()`, мы хотим, чтобы header-файлы указывались автоматически. Если до этого мы использавали заголовочные файлы только при компиляции исполняемого файла и нам хватало пути до build/ директории header файлов, сейчас же мы хотим, чтобы тот, кто использует нашу библиотеку получал путь до header файлов, но он допустим не билдит библиотеку, а просто скачал ее через установщик.
То есть когда мы используем `find_package()` нам не обязательно билдить саму библиотеку? Держать ее в проекте? Мы устанавливаем ее глобально и после можем использовать откуда угодно? Вроде бы так!

- Исполняемый файл - make, выполняется
- Архив - cpack
- Загрузочный файл - cpack
- Библиотека - устанавливается и дальше линкуется

```c
target_include_directories(MathFunctions INTERFACE
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
	$<INSTALL_INTERFACE:include/Tutorial>
)
...
install(
    FILES MathFunctions.h
    DESTINATION include/Tutorial
)
```

Все готово, но ...

---

Мы должны сгенерировать ==**_`MathFunctionsConfig.cmake`_**==, чтобы `find_package()` смог найти проект.

**_`Config.cmake.in`_** :
```c
@PACKAGE_INIT@
include ("${CMAKE_CURRENT_LIST_DIR}/MathFunctionsTargets.cmake")
```

`CMakeLists.txt` :
```c
include(CMakePackageConfigHelpers)
# as configure_file(), generates the config file that includes the exports
configure_package_config_file(
	# in
	${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
	# out
	"${CMAKE_CURRENT_BINARY_DIR}/MathFunctionsConfig.cmake"

	INSTALL_DESTINATION "lib/cmake/MathFunctions"
	# NO_SET_AND_CHECK_MACRO
	# NO_CHECK_REQUIRED_COMPONENTS_MACRO
)

# Дополнительно еще сгенерируем файл с версией
# и укажем backward compatibility
write_basic_package_version_file(
	"${CMAKE_CURRENT_BINARY_DIR}/MathFunctionsConfigVersion.cmake"
	VERSION "${Tutorial_VERSION_MAJOR}.${Tutorial_VERSION_MINOR}"
	COMPATIBILITY AnyNewerVersion
)

# Эти файлы должны находиться в стандартном расположении
install(
	FILES
		${CMAKE_CURRENT_BINARY_DIR}/MathFunctionsConfig.cmake
		${CMAKE_CURRENT_BINARY_DIR}/MathFunctionsConfigVersion.cmake
	DESTINATION lib/cmake/MathFunctions
)

# generate MathFunctionsTargets.cmake, allowing MathFunctionsConfig.cmake to be used without needing it to be installed
# 1. add_subdirectory
# 2. include()
export(
	EXPORT MathFunctionsTargets
	FILE "${CMAKE_CURRENT_BINARY_DIR}/MathFunctionsTargets.cmake"
)
```

`MathFunctionsConfig.cmake` -> `MathFunctionsTargets.cmake`
`MathFunctionsVersion.cmake`

Дальше после install, когда мы вызываем, мы ищем по названию папки, и все таргеты подтягиваются из библиотеки.

----

# Step 12. Packaging in Debug and Release

Здесь мы компилируем проект в Debug и Release, а потом зачем-то упаковываем их вместе в один установщик / архив.

В Debug файлы postfix-ятся буквой d, что странное решение со стороны туториала, потому что такое может значить daemon, как sshd.

Postfix d
```cpp
# Packaging Debug and Release
# postfix with d - libraries in debug
set(CMAKE_DEBUG_POSTFIX d)

add_subdirectory(MathFunctions)

# add the executable
add_executable(Tutorial tutorial.cxx)
# postfix with d - libraries in debug
set_target_properties(Tutorial PROPERTIES
	DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX}
)
```

```cpp
# Packaging Debug and Release
set_property(TARGET MathFunctions PROPERTY VERSION "1.0.0")
set_property(TARGET MathFunctions PROPERTY SOVERSION "1")
```

```c
include("../release/CPackConfig.cmake")

set(CPACK_INSTALL_CMAKE_PROJECTS
# Build directory, Project Name, Project Component, Directory
    "debug;Tutorial;ALL;/"
    "release;Tutorial;ALL;/"
# Stupidly все файлы генерятся в root директорию проекта
)

#$ mkdir debug release
#$ cd debug
#$ cmake -DCMAKE_BUILD_TYPE=Debug ..
#$ cmake --build .
#$ cd release
#$ cmake -DCMAKE_BUILD_TYPE=Release ..
#$ cmake --build .
#$
#$ cpack --config MultiCPackConfig.cmake

# Это скомпилирует проект в Debug и в Release.
# При Debug идет postfix d: MathFunctions.so -> MathFunctionsd.so
# После мы упаковываем и Debug, и Release вместе.
# Я не знаю зачем это делать, мб при тестировании?.
#
# Кстати, postfix d похож на daemon, я не уверен что это хороший postfix.
```
