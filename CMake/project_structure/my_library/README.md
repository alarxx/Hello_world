# Library structure

**Main Idea**  
The standard is public headers (interfaces) go in _`include/projname/`_,  
and everything else goes in _`src/`_ (or source, code).

**_`my_library/`_**
- _`build/`_
- _`include/my_library/`_ - public interfaces
	- _`math.h`_
- _`src/`_ - private
	- _`adder.cpp`_
	- _`adder.h`_
- _`CMakeLists.txt`_
- _`README.md`_
- _`.gitignore`_


**Usage:**
```cpp
#include "my_library/math.h"
// not #include "math.h"
```


**Example of Libraries:**
- assimp (3D): https://github.com/assimp/assimp
- poco(network): https://github.com/pocoproject/poco - sub-libraries
- glm (math): https://github.com/g-truc/glm - header only library
