#pragma once

/*
Оказывается, определение of templates должно быть доступно везде где их используют.
То есть на практике это значит, что нужно определение писать в header-файлах.

template <typename A, typename B>
auto add(A a, B b);

template <typename T, typename ... TArgs>
auto add(T a, TArgs ... args);
*/


// Base case
template <typename A, typename B>
auto add(A a, B b){
    return a + b;
}

// Variadic Templates capture a parameter pack
template <typename T, typename ... TArgs>
auto add(T a, TArgs ... args){ // pack expansion by ellipsis(...)
    // int values[] = {args...}; // arguments expansion
    return a + add(args...);
}
