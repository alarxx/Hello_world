#include <iostream>
#include <string>
#include <vector>
// SFINAE
#include <utility> // std::declval<T> - rvalue of class without creation of the class
#include <type_traits>
// decltype without std:: ?
// integral_constant: true_type, false_type
// void_t ? <any> -> void


// --- enable_if ---
template <bool Condition, typename T = void>
class enable_if {
// type is not defined of Condition == false
};

template <typename T>
class enable_if<true, T> {
public:
	using type = T;
};

// Alias of enable_if<>::type
template <bool Condition, typename T = void>
using enable_if_t = typename enable_if<Condition, T>::type;
// ------


// --- fun ---
template <typename T>
enable_if_t<std::is_floating_point_v<T>, float>
    fun(T t){
        std::cout << "float fun: " << typeid(T).name() << std::endl;
        return t;
    }

template <typename T>
enable_if_t<std::is_integral_v<T>, int>
    fun(T t){
        std::cout << "int fun: " << typeid(T).name() << std::endl;
        return t;
    }
// ------


/*
    Как написать свою проверку?
    has_size()
 */
template <typename ... >
using void_t = void;

template <typename T, T v>
class integral_constant {
public:
    static constexpr T value = v;
    using value_type = T;
    using type = integral_constant<T, v>;
    constexpr operator T(){ return v; }
};
// template <typename T>
// class true_type : public integral_constant<T, true> {};
using true_type = integral_constant<bool, true>;
// template <typename T>
// class false_type : public integral_constant<T, false> {};
using false_type = integral_constant<bool, false>;
// ------


// --- has_size ---
// Base
template <typename T, typename U = void>
// class has_size : public false_type {};
class has_size {
public:
    static constexpr bool value = false;
};

// Template Partial Specialization
template <typename T>
// class has_size< T, decltype(std::declval<T>().size(), void()) > : public std::true_type {}; // OK, comma operator
// class has_size< T, void_t<decltype(std::declval<T>().size())> > : public true_type {};
class has_size< T, void_t<decltype(std::declval<T>().size())> > {
public:
    static constexpr bool value = true;
};

// Alias of has_size<T>::value
template <typename T>
inline constexpr bool has_size_v = has_size<T>::value;
// ------

template <typename T>
enable_if_t<has_size_v<T>, T>
    fun(T t){
        std::cout << "has size fun: " << typeid(T).name() << std::endl;
        return t;
    }


int main(){
    enable_if_t<2+2==4, int> a = (float) 42.f;
    std::cout << a << ": " << typeid(a).name() << std::endl;

    fun(1);
    fun(1.f);
    fun('A');
    fun(false);

    fun(std::string("Hello"));

    std::vector veci = {1, 2, 3, 4};
    // fun(std::vector({1, 2, 3}));
    // fun(std::vector<int>());
    fun(veci);

    std::vector<float> vecf = {1, 2, 3, 4};
    // std::vector vecf = {1.f, 2.f, 3.f, 4.f};
    fun(vecf);

}
