#include <iostream>

#include "recursive_add.hpp"

template <typename T>
auto square(T t){
    return t * t;
}

template <typename ... TArgs>
auto add_square(TArgs ... args){
    return add(square(args)...);
}

int main(){
    auto res = add_square(1, 2.f, 3.); // 1 + 4 + 9 = 14
    std::cout << res << typeid(res).name() << std::endl;
}

