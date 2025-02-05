#pragma once

#include <iostream>

// Declaration, `extern` to show external relations
// otherwise multiple declaration
const int PI = 3.14;
extern int FOOBAR;
extern int FOOBAR;
// functions are extern by default (have global scope).

// Functions in headers must be inline,
// otherwise there will be multiple definitions

// function pointers
// inline function
// higher order function
inline int hof(int (*fun_ptr)(const int x)){
    return fun_ptr(this);
}

// must be inline, otherwise there will be multiple definitions
inline int inline_fun(){
    std::cout << "Inline Function!" << std::endl;
}

int global();

namespace mymath::myadd {
    int add(int a, int b);
    float add(float a, float b);
    /**
     * Множественное определение mymath::myadd

     int add(int a, int b){ return a + b; }
     float add(float a, float b){ return a + b; }

     */
}

namespace mymath::myadd {
    int add(int a, int b);
    float add(float a, float b);
}

int mymath::myadd::add(int a, int b){ return a + b; }
float mymath::myadd::add(float a, float b){ return a + b; }
