#include <iostream>
#include <type_traits>

template <typename T>
concept Arithmetic = std::is_arithmetic_v<T>;

/*
 define a function to sum only arithmetic types
 */
// Кажется нельзя создавать специализации для функций
template <Arithmetic A, Arithmetic B>
auto add(A a, B b){
    return a + b;
}

auto addc(Arithmetic auto a, Arithmetic auto b){
    return a + b;
}

class A {}; // not arithmetic

int main(){
    auto sum = addc(1, 2.5);
    std::cout << sum << typeid(sum).name() << std::endl;
    // std::cout << add(A{}, A{}) << std::endl; // error: no match for ‘operator+’
}
