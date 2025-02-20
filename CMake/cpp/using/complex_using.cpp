#include <iostream>

using std::cout, std::endl;

void f1(int a){ cout << "f1: " << a << endl; }
void f2(int a){ cout << "f2: " << a << endl; }
void f3(int a){ cout << "f3: " << a << endl; }

class MyClass {
public:
    void print(int a){ cout << "MyClass::print: " << a << endl; }
};

#include <functional> // обертка для функций, lambda, методов
using Callback = std::function<void(int)>;
// using Callback = void(*)(int);
void fun(int a){
    cout << "fun: " << a << endl;
}

int main(){
    using ActionArray = void(*[3])(int); // массив указателей на функции возвращающие void

    // void(* arrs[3])(int args) = {f1, f2, f3};
    ActionArray arrs = {f1, f2, f3};
    arrs[0](5);
    arrs[1](6);
    arrs[2](7);

    MyClass obj;
    void (MyClass::*ptr)(int) = &MyClass::print;
    (obj.*ptr)(42);

    Callback cb = fun;
    cb(123);
}
