# Library structure

**Main Idea**  
The standard is public headers (interfaces) go in _include/projname/_,
and everything else goes in src (or source, code).

**_my_library/_**
- _build/_
- _include/my_library/_ - public interfaces
	- _math.h_
- _src/_ - private
	- _adder.cpp_
	- _adder.h_
- _CMakeLists.txt_
- _README.md_
- _.gitignore_


**Usage:**
```cpp
#include "my_library/math.h"
// not #include "math.h"
```


**Example of Libraries:**
- assimp (3D): https://github.com/assimp/assimp
- poco(network): https://github.com/pocoproject/poco - sub-libraries
- glm (math): https://github.com/g-truc/glm - header only library
