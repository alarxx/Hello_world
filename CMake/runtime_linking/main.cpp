#include <iostream>

// Dynamic Linking
// I guess it means Dynamic Library Functions?
#include <dlfcn.h>

int main(){
    void * handle = dlopen("./mymathlib/libmymath.so", RTLD_LAZY);
    if(!handle){
        std::cerr << "Failed to load library: " << dlerror() << std::endl;
        return 1;
    }

    using add_t = int (*)(int, int);
    using sub_t = int (*)(int, int);

    // dlsym() search for the named symbol in all objects loaded automatically
    // as a result of loading the object referenced by handle
    // check name mangling in call_c_from_cpp
    add_t add = (add_t) dlsym(handle, "add");
    add_t sub = (add_t) dlsym(handle, "sub");

    if(!add || !sub){
        std::cerr << "Failed to load functions: " << dlerror() << std::endl;
        dlclose(handle);
        return 1;
    }

    std::cout << "1 + 1 = " << add(1, 1) << std::endl;
    std::cout << "1 - 1 = " << sub(1, 1) << std::endl;

    dlclose(handle);
    return 0;
}

// g++ main.cpp -o app -ldl
