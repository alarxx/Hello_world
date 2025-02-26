#include <iostream>
#include <type_traits>

template <typename T, std::enable_if_t<!std::is_arithmetic_v<T>, int> = 0>
constexpr bool is_arithmetic(){
    return false;
}
template <typename T, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
constexpr bool is_arithmetic(){
    return true;
}

template <typename T>
concept Arithmetic = std::is_arithmetic_v<T> && is_arithmetic<T>();
// Почему-то он пишет, что statements must be primary expression, и не могут быть constexpr function

// It acts like SFINAE
template <Arithmetic A, typename B>
requires Arithmetic<B>
auto add(A a, B b) requires (sizeof(B) >= 4) {
    return a + b;
}

int main(){
    auto sum = add(1, 2.5);
    std::cout << sum << typeid(sum).name() << std::endl;
}
