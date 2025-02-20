#include <iostream>

// typedef и using используются для aliases (псевдонимы)
// Aliases можно сделать только для типов, но не переменных или объектов.

// using используется для namespace-ов и template-ов
using std::cout, std::endl;
// using cout = std::cout; // Error
// Можно еще использовать препроцессор
#define out std::cout

// Template Aliase for any pointer
// typedef
template <typename T> class Container {
public:
    typedef T* Ptr;
};
// using
template <typename T> using Ptr = T*;

// Function to point from function aliases
void fun(int a){
    std::cout << "Value: " << a << std::endl;
}

int main(){

    // Aliases
    // typedef int MyInt;
    using MyInt = int;

    MyInt a = 123;
    cout << a << endl;

    // ISO C++ forbids converting a string to char*
    using StringPtr = char *;
    StringPtr c = "Hello, World!";
    cout << c << endl;


    // typedef int IntArray[5];
    using IntArray = int[5];
    IntArray arr;
    for(int i=0; i<5; i++){
        arr[i] = i;
    }
    for(int i=0; i<5; i++){
        cout << arr[i] << " ";
    }
    cout << endl;


    // Container<int>::Ptr p = new int(10);
    Ptr<int> p = new int(10);
    cout << *p << endl;
    delete p;

    using FunPtr = void(*)(int);
    // void (*funptr)(int) = fun;
    FunPtr funptr = fun;
    funptr(10);

}
