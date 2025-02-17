/*
 * Шаблонный класс принимающий в виде типа шаблонный класс
 */

#include <iostream>

// Получается это не класс, а шаблон
template <typename T> class A {
public:
    T data;
    // Initializer list
    A(T data) : data(data){}
};

// template <template <typename> typename T, typename S = char> class B { // invalid!
template <template <typename> class T, typename S = char> class B { // valid!
// template <class T, typename S> class B { // Compilation Error! Because we use T as template by itself
public:
    T<S> obj; // A<int>
    B(S value) : obj(value) {} // B(int value){ obj(value); }
    S get(){
        return obj.data;
    }
};

int main(){
    B<A, int> obj(10);

    auto value = obj.get();
    std::cout << value << std::endl;
}
