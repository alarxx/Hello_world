# concepts

C++20

Concepts allows compile-time validation of template argument, as [[Substitution Failure Is Not An Error (SFINAE)|SFINAE]].

SFINAE solution is ugly, verbose:
```cpp
// enable_if<bool, type>::type
template <typename T, enable_if<is_arithmetic_v<T>, int>::type = 0>
void fun(T t){}
```

**Concepts** это compile-time проверка/ограничения/constraints, чтобы одобрить только определенные типы в template-ах.

- `concept`
- `requires` clause
- `requires` expression

---
####  `concept` and `requires` clause

`concept`:
```cpp
#include <type_traits>
#include <concepts>

template <typename T>
concept Arithmetic = std::is_arithmetic_v<T>; // bool

template <Arithmetic A, Arithmetic B>
requires (sizeof(A) >= 4) // clause acts like SFINAE
auto add(A a, B b) /*requires clause can also be here*/ {
	return a + b;
}

int main(){
	auto sum = add(1 + 2.5);
	// sum = 3.5
	// typeid(sum).name() = d
}
```

---
#### `requires` expression

`requires` expression:
```cpp
requires [(arguments)] {
	[SFINAE constrain];
	requires [predicate];
} -> bool;
```

```cpp
#include <iostream>
#include <concepts>

template <typename T>
concept Arithmetic = std::is_arithmetic_v<T>;
// Почему-то он пишет, что statements must be primary expression, и не могут быть constexpr function
// Хотя они могут...

template <typename T>
concept MyConcept = requires (T a, T b){
    // SFINAE constrains
    typename T::type;
    a.x;
    a.f();
    a[0];
    a + b;
    {*a + 1} -> std::convertible_to<float>;
    {a * a} -> std::same_as<int>;
    // requires clause
    requires std::is_arithmetic_v<T>; // SFINAE
    requires Arithmetic<T>; // concept
	requires requires (T t) { t.x; }; // Nested requires expressions
};

template <MyConcept T>
requires requires (T t) { t.x; } // clause combined with expression
void fun(T t) requires requires (T t) { t.x; } {
    std::cout << "okay" << std::endl;
}

class A {
public:
    using type = int;
    int x;
    A() = default;
    A(int x) : x(x) {}
    void f(){};
    int operator [] (const int i) const {
        return i;
    }
    friend int operator + (const A& a1, const A& a2);
    friend int operator * (const A& a1, const A& a2);
    // same_as<int> требует, чтобы без casting-а было int уже
    // friend A operator * (const A& a1, const A& a2);
    operator int () const { return x; }
    int& operator * (){ return x; }
};

// Specialization of std::is_arithmetic for A
// I'd not recommend to do it
template <>
class std::is_arithmetic<A> : public std::true_type {};

int main(){
    A a(42);
    fun(a);

    int num = static_cast<int>(a);
    std::cout << num << std::endl;
}
```

---

#### `requires` and `constexpr`

```cpp
template <typename T>
constexpr bool has_member_x = requires(T v){ v.x };
```

```cpp
if constexpr (MyConcept<T>) ?
```

```cpp
static_assert(requires(T v){ ++v ;}, "no increment")
```

```cpp
template <typename Iter>
constexpr bool is_iterator(){
	return requires(Iter it){ *it++; };
}
```
