# Simple Library

Здесь мы будем build-ить static library и link-овать её в приложении:
1) Build _`mymathlib/`_
2) You can also "install" library
3) Link the library in _`application/`_

---

В общем, мы можем скопировать и сбилдить библиотеку сами, либо в случае с closed source нам обычно предоставляют pre-built shared library и headers, и после можно link with your project.

Для понимания использования библиотек полезно знать процесс компиляции.

---

## Additionally about Compilation Process

Short video: https://www.youtube.com/shorts/B2O5uk_0leA

Reference: https://www3.ntu.edu.sg/home/ehchua/programming/cpp/gcc_make.html

![Pasted image 20250123153253](https://github.com/user-attachments/assets/e37d4b57-cdf9-4fc8-a20d-92087292ddef)

`.cpp`+`.h` -> `.i` -> `.asm|.s` -> `.obj|.o` -> `.exe|.a|.so`

Препроцессинг делает "pure" C++.  
Этот код потом компилируется в ассембли.  
Ассембли ассемблируется в машинный код.  
Машинный код объединяется в один исполняемый файл или библиотеку - Linking.  


### Static and Dynamic Linking

- **Static Libraries:** 
	- .a (archive file) - linux
	- .lib (library) - windows
- **Shared Libraries:** 
	- .so (shared objects) - linux
	- .dll (dynamic link library) - windows
- **Header-only library** (static)

Dynamic linking круче, потому что не создается копия внутри executable и значительно экономится память, когда например несколько программ используют одну и ту же библитеку.

GCC по-умолчанию линкует shared libraries.

**Searching headers and libraries**
- `-Idir`: (_include-paths_) где искать header-files of `#include`
- `-Ldir`: (_library-paths_) где искать библиотеки
- `-lxxx` and `-lxxx.lib`: for libraries itself

GCC Environment Variables:
- `PATH` searching for .dll, .so
- `CPATH` - _include-paths_, searches after `-Idir`
	- `C_INCLUDE_PATH`
	- `CPLUS_INCLUDE_PATH`
- `LIBRARY_PATH` - _library-paths_, searches after `-Ldir`
