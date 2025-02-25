### Variadic Templates

[[federico - Modern CPP Programming]]

**_`recursive_add.hpp`_**
```cpp
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
```

---

**Function Unpack**

```cpp
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
```

---

**Function Application**

```cpp
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
```

---
#### Tuple

```cpp
#include <iostream>

template <typename ... TArgs>
class Tuple;

template <typename T>
class Tuple<T> { // Base Case
public:
    T value;
    Tuple(T value) : value(value) {}
};

template <typename T, typename ... TArgs>
class Tuple<T, TArgs...> {
public:
    T value;
    Tuple<TArgs...> tail;
    Tuple(T value, TArgs ... args) : value(value), tail(args...) {}
};

// Deduction Guide to Support Type Deduction
template <typename ... TArgs>
Tuple(TArgs ...) -> Tuple<TArgs...>;

int main(){
    // Tuple tuple(2, 2.0, 'a');
    Tuple tuple = {2, 2.0, 'a'};

    std::cout << tuple.value << std::endl;
    std::cout << tuple.tail.value << std::endl;
    std::cout << tuple.tail.tail.value << std::endl;
}
```

