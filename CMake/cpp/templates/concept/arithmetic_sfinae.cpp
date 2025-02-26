#include <iostream>
#include <type_traits>

/*
 define a function to sum only arithmetic types
 */
// Кажется нельзя создавать специализации для функций
template <typename A, typename B, std::enable_if_t<std::is_arithmetic_v<A> && std::is_arithmetic_v<B>, int> = 0>
auto add(A a, B b){
    return a + b;
}

class A {}; // not arithmetic

int main(){
    auto sum = add(1, 2.5);
    std::cout << sum << typeid(sum).name() << std::endl;
    // std::cout << add(A{}, A{}) << std::endl; // error: no match for ‘operator+’
}
