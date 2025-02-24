/*
https://github.com/federico-busato/Modern-CPP-Programming/issues/181

Вы пишите что невозможно вызвать (1), если (2) существует, и поэтому я думаю что параметрах of template не должно быть int Size, иначе все идеально работает.
Дело снова в дедукции типа, если явно не указать тип и Size, то эта функция не будет выбираться компилятором.
 */

#include <iostream>
#include <type_traits>

// --- reference ---
template <typename T, int Size>
void
f(T (& t)[Size]){
    std::cout << "&[" << Size << "]: " << typeid(t).name() << std::endl;
} // (1)
// ------

// --- pointer ---
template <typename T, int Size>
void
f(T * t){
    std::cout << "*: " << typeid(t).name() << std::endl;
} // (2)
// ------

// --- base ---
template <typename T>
std::enable_if_t<std::is_pointer_v<T>, void>
f(T t){
    std::cout << "base: " << typeid(t).name() << std::endl;
} // (3)
// ------

int main(){
    int array[3];
    f(array);   // (1) int(&)[3] - array reference
    f(&array);  // (3) int(*)[3] - array pointer

    int * array_ptr = new int[3];
    f(array_ptr);           // (3) base int*
    f<int, 3>(array_ptr);   // (2) int* <-- main point
}
