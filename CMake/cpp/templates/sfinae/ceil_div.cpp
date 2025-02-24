/*
 What is SFINAE?
 Шаблоны которые не подходят под переданный тип исключаются и переходят на следующий, а не выкидывают ошибку.

 В примере ниже мы declare template функцию,
 потом определяем specialization для int и unsigned int,
 И дедукция работате для этих функций, но проблема возникнет, когда мы заходим передать long, long long, float, double, etc.
 */

#include <iostream>
using std::cout, std::endl;

template <typename T>
T ceil_div(T value, T div);

template <>
unsigned ceil_div<unsigned>(unsigned value, unsigned div){
    return (value + div - 1) / div;
}

template <>
int ceil_div<int>(int value, int div){
    // XOR bitwise operator
    // Если один из них меньше 0, то результат будет с минусом: (-5/2)=-2.5; ceil(-2.5)=-2
    return (value > 0) ^ (div > 0) ? (value / div) : (value + div - 1) / div;
}

int main(){
    int c = ceil_div(8, 2); // Ok
    cout << c << ": " << typeid(c).name() << endl;

    unsigned u = 10;
    cout << u << ": " << typeid(u).name() << endl;

    unsigned uu = ceil_div(8u, 2u); // Ok
    // unsigned long luu = ceil_div(8lu, 2lu); // Compilation Error: undefined reference to `unsigned long ceil_div<unsigned long>(unsigned long, unsigned long)'
    unsigned long luu = ceil_div<int>(8lu, 2lu); // Ok, without deduction, explicitly setting to int
}
