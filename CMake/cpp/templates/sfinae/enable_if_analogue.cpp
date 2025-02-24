#include <iostream>
#include <vector>
#include <string>
#include <type_traits>

template <typename T>
struct has_size {
private:
    // Trailing return type: auto fun() -> int;
    template <typename U> // without it it won't work
    static constexpr auto test(int) -> decltype(std::declval<U>().size(), std::true_type());
    // static constexpr auto test(int) -> decltype(std::declval<T>().size(), std::true_type());

    /*
     SFINAE работает только в случае подстановки шаблонного параметра.
     То есть это нужно чтобы компилятор мог выбрать какой из шаблонов выбрать.
     */

    template <typename>
    static constexpr std::false_type test(...);
public:
    static constexpr bool value = decltype(test<T>(0))::value;
    // static constexpr bool value = decltype(test(0))::value;
};

// template <typename T>
// using has_size = decltype(std::declval<T>().size(), std::true_type);


int main() {
    std::cout << std::boolalpha;
    std::cout << "std::string has size(): " << has_size<std::string>::value << std::endl;  // true
    std::cout << "std::vector<int> has size(): " << has_size<std::vector<int>>::value << std::endl; // true
    std::cout << "int has size(): " << has_size<int>::value << std::endl;  // false
}

