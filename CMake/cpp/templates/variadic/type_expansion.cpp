#include <iostream>

template <typename A, typename B>
auto g(A a, B b){
    std::cout << "g (base), \n\ta:" << a << typeid(a).name()
        << ", \n\tb: " << b << typeid(b).name() <<  std::endl;
    return a + b;
}

template <typename A, typename ... TArgs>
auto g(A a, TArgs ... args){
    std::cout << "g, a:" << a << typeid(a).name() << std::endl;
    return a + g(args...);
}

template <typename ... TArgs>
auto f(TArgs ... args){
    return g<std::make_unsigned_t<TArgs>...>(args...);
}

int main(){
    // std::cout << typeid(std::make_unsigned_t<int>).name() << std::endl;  // "unsigned int"
    // std::cout << typeid(std::make_unsigned_t<short>).name() << std::endl; // "unsigned short"
    // std::cout << typeid(std::make_unsigned_t<char>).name() << std::endl;  // "unsigned char"

    unsigned int j = (unsigned int) -1;
    std::cout << j << typeid(j).name() << std::endl;  // "unsigned int"
    // 4294967295j - max unsigned number
    // Two's Compliment

    auto y = f(1, 2, 3);
    std::cout << y << typeid(y).name() << std::endl;
}
