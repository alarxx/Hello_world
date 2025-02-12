#include <iostream>
#include <cassert>

#include "legacy.h"
// C declaration without name mangling

int main(){
    assert(f() == 1 && "f() should be 1");
    std::cout << "This is fine." << std::endl;
}
