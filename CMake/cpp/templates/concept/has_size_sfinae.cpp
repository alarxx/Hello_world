#include <iostream>
#include <string>

template <typename T, T v>
class integral_constant {
public:
    using value_type = T;
    static constexpr T value = v;
};

class true_type : public integral_constant<bool, true> {};
class false_type : public integral_constant<bool, false> {};


template <bool Statement, typename T>
class enable_if {};

template <typename T>
class enable_if<true, T> {
public:
    using type = T;
};

template <bool Statement, typename T>
using enable_if_t = typename enable_if<Statement, T>::type;


template <typename ... T>
using void_t = void;

template <typename T, typename = void>
class has_size : public false_type {};

template <typename T>
class has_size<T, void_t<decltype((void) std::declval<T>().size(), void())>> : public true_type {};
// Получается есть 3 способа обернуть decltype: void_t<>, comma operator, and casting

template <typename T>
constexpr bool has_size_v = has_size<T>::value;

template <typename T, enable_if_t<has_size_v<T>, int> = 0>
void fun(T t){
    std::cout << t.size() << ": " << t << std::endl;
}

// To prevent error
template <typename T, enable_if_t<!has_size_v<T>, int> = 0>
void fun(T t){
    std::cout << "no size" << std::endl;
}

int main(){
    std::string str = "Hello";
    fun(str);

    int a = 42; // error: no matching function for call to ‘fun(int&)’
    fun(a);
}
