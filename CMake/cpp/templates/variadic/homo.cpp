#include <iostream>
#include <concepts>
#include <type_traits>

// template <int ... IntSeq>
// void f(IntSeq ... seq){} // Error!
// https://github.com/federico-busato/Modern-CPP-Programming/issues/183

// For functions you can use variadic templates with SFINAE type check
// or with requires (C++20) or as it is written below with auto, it seems to be called "abbreviated function templates"

// I believe this solution should work below С++20:
// Pack expansion со SFINAE type check for all to be int
// `conjunction` performing a logical AND, references:
// https://en.cppreference.com/w/cpp/types/conjunction
// https://en.cppreference.com/w/cpp/types/conditional
template <typename ... TArgs, typename = std::enable_if_t<(std::conjunction_v<std::is_same<TArgs, int>...>)>>
void fun(const TArgs ... args) {}

// abbreviated lambda function templates
auto lambda = [](auto ... args){};

// constrained abbreviated function templates
void fun(std::integral auto ... args){
    int arr[] = {args...};
    std::cout << sizeof(arr) << std::endl;
}

int main(){
    fun();
}
