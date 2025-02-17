#include <iostream>
#include <initializer_list>

using namespace std;

class A {
};

int main() {
    // int arr[0]; // Error!
    int arr[3]{1, 2,};

    // Nice Syntax with initializer_list
    // 1. Uniform Initialization        :   arr{1, 2, 3, 4, 5}
    //    Actually, we call constructor :   arr({1, 2, 3, 4, 5})
    // 2. Here we use Copy Operator
    // {x, y, z} in Stack
    // By default initializer_list don't implement deep copy, so (initializer_list={x, y}) is local only
    // Оказывается, std::vector при присваивании создает копию.
    // В имплементации приходится создавать копию, потому что initializer_list в Stack памяти.

    initializer_list<int> list = {10, 20, 30, 40, 50};
    // for (auto i : list) {
    //     cout << i << "\n";
    // }

    int * array = new int [list.size()];
    // std::cout << list.begin() << std::endl; // pointer
     // ---
    // const int * array = list.begin(); // мы могли бы так сделать, но ...
    // 1) сами значения array [i] были бы const
    // 2) initializer_list in Static Memory, so it will be deleted after function completion what will lead to dangling pointer
    // ---
    // So we create copy
    int i = 0;
    // typename initializer_list<int>::iterator
    // auto
    // for(auto it = list.begin(); it != list.end(); it++, i++){
    // std::initializer_list<T>::iterator // Тут почему-то нельзя использовать из-за шаблонного initializer_list
    // typename initializer_list<int>::iterator
    for (initializer_list<int>::iterator it = list.begin(); it != list.end() && i < list.size(); it++, i++) {
        array[i] = *it;
        cout << array[i] << "\n";
    }


    delete[] array;

    return 0;
}

