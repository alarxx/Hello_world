/*
 Template класс хранит его тип в поле, к которому можно получить доступ
 */
#include <iostream>

template <typename T>
class A {
public:
    using type = T;
};

template <typename T> using AType = typename A<T>::type;

template <typename T>
void fun(A<T> a){
    AType<T> var; // typename A<T>::type var;
    std::cout << typeid(var).name() << std::endl;
}

int main(){
    A<int> ai;

    std::cout << typeid(ai).name() << std::endl;

    fun(ai);
}
