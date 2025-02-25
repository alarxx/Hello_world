#include <iostream>

template <typename ... TArgs>
void f(TArgs ... args){
    std::cout << "f" << std::endl;
} // pass by value

template <typename ... TArgs>
void g(const TArgs& ... args){
    std::cout << "g" << std::endl;
} // pass by const reference

template <typename ... TArgs>
void h(TArgs* ... args){
    std::cout << "h" << std::endl;
} // pass by pointer

template <int ... Sizes>
void l(int (&...arrays)[Sizes]){
    std::cout << "l" << std::endl;
} // pass a list of array references

int main(){
    int a[] = {1, 2};
    int b[] = {1, 2, 3};
    f(1, 2.0);
    h(a, b);
    l(a, b);
    g(a, b);
}
