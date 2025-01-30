# Simple Library

1) Build library
2) Link library in application

---

**Example of Libraries:**
- assimp (3D): https://github.com/assimp/assimp
- poco(network): https://github.com/pocoproject/poco - sub-libraries
- glm (math): https://github.com/g-truc/glm - header only library

**Library structure**

The standard is public headers (interfaces) go in _include/projname/_,
and everything else goes in src (or source, code).

_my_library/_
- _build/_
- _include/my_library/_ - public interfaces
	- _math.h_
- _src/_ - private
	- _adder.cpp_
	- _adder.h_
- _CMakeLists.txt_
- _README.md_
- _.gitignore_


Usage:
```cpp
#include "my_library/math.h"
// not #include "math.h"
```

---

**Application structure**

Options:
- folders per class
- folders per section
- pure flat
- headers for everything
	- one main cpp file
	- single object rebuild flaw (long compile time for a single change)
- monolithic - "thick", i.e. all project in one main.cpp

Folder per class:
_application/_
- _Data/_
	- _Data.cpp_
	- _Data.h_
- _main.cpp_

---


**Including Libraries**

- use manager:
	- [[conan]]
	- [[vcpkg]]
- do it manually
	- download, build, link
		- or download prebuilt and link
	- git submodules

---

**Build Systems**

- [[CMake]]
- premake?

## References:
- 30m C++ Crash Course! The Best Project Setups for Libs and Apps. Also Standards, and More! https://www.youtube.com/watch?v=7KAhreWsIQI.
