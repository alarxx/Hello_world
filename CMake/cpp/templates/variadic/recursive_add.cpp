#include <iostream>

#include "recursive_add.hpp"

int main(){
    short short_num = (short) 1;
    auto sum = add(
        short_num,
        1u,
        1,
        1l,
        1.f,
        1.,
        1.l
    );
    std::cout << sum << typeid(sum).name() << std::endl; // 7e - long double

    // add(1); // Compile Error
}
