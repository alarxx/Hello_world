#include <iostream>
#include <type_traits>


// --- base ---
template <typename T>
std::enable_if_t<std::is_pointer_v<T>, void>
f(T t){
    std::cout << "base: " << typeid(t).name() << std::endl;
} // (3)
// ------


// --- pointer ---
template <typename T, int Size>
void
f(T (*t)[Size]){
    std::cout << "*[" << Size << "]: " << typeid(t).name() << std::endl;
}

template <typename T, int Size>
void
f(T * t){
    std::cout << "*: " << typeid(t).name() << std::endl;
} // (2)
// ------


// --- reference ---
template <typename T, int Size>
void
f(T (& t)[Size]){
    std::cout << "&[" << Size << "]: " << typeid(t).name() << std::endl;
} // (1)

template <typename T, int Size>
void
f(T & t){
    std::cout << "&: " << typeid(t).name() << std::endl;
}
// ------


// --- rvalue ---
template <typename T, int Size>
void
f(T (&& t)[Size]){
    std::cout << "&&[" << Size << "]: " << typeid(t).name() << std::endl;
}

template <typename T, int Size>
void
f(T && t){
    std::cout << "&&" << typeid(t).name() << std::endl;
}
// ------


int main(){
    int array[3];
    f(array); // int(&)[3] - array reference
    f(&array); // int(*)[3] - array pointer

    f(std::move(array)); // int(&&)[3] - array rvalue
    f({1, 2, 3}); // int(&&)[3]

    f(new int[3]); // base int* - pointer

    int * array_ptr = new int[3];
    f(array_ptr); // base int*
    f<int, 3>(array_ptr); // int* - specialization

    f(&array_ptr); // int**
    f<int*, 3>(&array_ptr); // int**
}
