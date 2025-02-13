#include <iostream>

// `extern "C"` заставляет компилятор C++ использовать чистое имя, без name mangling,
// поэтому мы можем искать функцию через `dlsym(handle, "add");`
// check name mangling in call_c_from_cpp
extern "C" {
    int add(int a, int b){ return a + b; }
    int sub(int a, int b){ return a - b; }
}
