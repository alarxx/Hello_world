#include <iostream>
#include <concepts>

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

/*
 requires [(arguments)] {
    [SFINAE constrain];
    requires [predicate];
 } -> bool;
 */
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
requires requires (T t) { t.x; }
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
