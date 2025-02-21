
#include <iostream>

using std::cout, std::endl;

// fun принимает только временные объекты (rvalue) и привязывает к a, при этом создавая для a memory address
void fun(int && a){
    // a - lvalue уже, но мы передаем сюда rvalue
    cout << "&&arg: " << a << " addr(" << &a << ")" << endl;
}
void fun(int & a){
    // a - lvalue уже, но мы передаем сюда rvalue
    cout << "&arg: " << a << " addr(" << &a << ")" << endl;
}

int main(){

    int a = 15; // a - lvalue
    cout << "a: " << a << " addr(" << &a << ")" << endl;

    fun(a); // call-by-reference

    fun(10);
    fun(int(a)); //  explicitly создаем копию, потому что fun принимает только rvalue

    int b = -1;
    fun(std::move(a)); // "относись как к rvalue"
    int c = -1;
    cout << "a: " << a << " addr(" << &a << ")" << endl;
}

