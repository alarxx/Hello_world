/*
 * Здесь я создаю 2 шаблонные функции с проверкой на has_size и is_arithmetic.
 * Проверку типа можно написать в:
 * - Hidden Template Parameter
 * - Hidden Function Parameter
 * - Return Type
 */
#include <iostream>
#include <type_traits>
#include <string>

template <typename T, typename U = void>
class has_size : public std::false_type {};

// std::void_t or using Comma Operator no difference, it is still just void
// template <typename T>
// class has_size<T, decltype(std::declval<T>().size(), void())> : public std::true_type {};
template <typename T>
class has_size<T, std::void_t<decltype(std::declval<T>().size())>> : public std::true_type {};

template <typename T>
constexpr bool has_size_v = has_size<T>::value;


/*
// --- SFINAE in Function as Main Parameter ---
// This somehow doesn't work
// https://github.com/federico-busato/Modern-CPP-Programming/issues/180
template <typename T>
void fun(std::enable_if_t<std::is_arithmetic_v<T>, T> t){
    std::cout << "is_arithmetic: " << t << ", type: " << typeid(t).name() << std::endl;
}
template <typename T>
void fun(std::enable_if_t<has_size_v<T>, T> t){
    std::cout << "has_size: " << t << ", type: " << typeid(t).name() << std::endl;
}
// ------
*/

/*
// --- Hidden Function Parameter ---
template <typename T>
void fun(T t, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0){
    std::cout << "is_arithmetic: " << t << ", type: " << typeid(t).name() << std::endl;
}
template <typename T>
void fun(T t, std::enable_if_t<has_size_v<T>, int> = 0){
    std::cout << "has_size: " << t << ", type: " << typeid(t).name() << std::endl;
}
// ------
*/

/*
// --- Hidden Template Parameter (IMO the best) ---
template <typename T, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
void fun(T t){
    std::cout << "is_arithmetic: " << t << ", type: " << typeid(t).name() << std::endl;
}
template <typename T, std::enable_if_t<has_size_v<T>, int> = 0>
void fun(T t){
    std::cout << "has_size: " << t << ", type: " << typeid(t).name() << std::endl;
}
// ------
*/

// --- SFINAE as return type ---
template <typename T>
std::enable_if_t<std::is_arithmetic_v<T>, void>
fun(T t){
    std::cout << "is_arithmetic: " << t << ", type: " << typeid(t).name() << std::endl;
}
template <typename T>
std::enable_if_t<has_size_v<T>, void>
fun(T t){
    std::cout << "has_size: " << t << ", type: " << typeid(t).name() << std::endl;
}
// ------

int main(){
    int ai = 42;
    fun(ai);

    std::string str = "Hello";
    fun(str);
}
