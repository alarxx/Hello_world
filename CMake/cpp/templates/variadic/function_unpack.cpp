#include <iostream>

#include "recursive_add.hpp"

// // Specialization
// template<> auto add(int a, int b){
//     std::cout << "Specialization (base)" << std::endl;
//     return a + b;
// }
// template<typename ... TArgs> auto add(int a, TArgs ... args){
//     std::cout << "Specialization" << std::endl;
//     return a + add(args...);
// }

struct A {
    int v;
    int f(){ return v; }
};

template <typename ... TArgs>
auto f(TArgs ... args){
    return add(args.f()...); // args[0].f() + args[1].f() + args[2].f()
}

int main(){
    // Check Specialization
    // std::cout << add(1, 1) << std::endl;

    auto res = f(A{1}, A{2}, A{3});
    std::cout << res << std::endl;
}
