#include <iostream>

using std::cout, std::endl;

int main(){
    int _b = 10;
    // _b: 10 addr(0x7ffe2f3038ac)
    cout << "_b: " << _b << " addr(" << &_b << ")" << endl;

    int * b = &_b;
    // *b: 10, b: 0x7ffe2f3038ac, &b: 0x7ffe2f3038a0

    // regular reference variable
    int & lvalue = *b; // copy constructor
    // lvalue: 10 addr(0x7ffe2f3038ac)

    cout << "*b: " << *b << ", b: " << b << ", &b: " << &b << endl;
    cout << "lvalue: " << lvalue << " addr(" << &lvalue << ")" << endl;

    cout << "---" << endl;

    int _a = 5;
    // _a: 5 addr(0x7ffe2f30389c)
    cout << "_a: " << _a << " addr(" << &_a << ")" << endl;
    int * a = &_a;
    // *a: 5, a: 0x7ffe2f30389c, &a: 0x7ffe2f303890
    cout << "*a: " << *a << ", a: " << a << ", &a: " << &a << endl;

    lvalue = *a; // copy assignment operator
    // lvalue: 5 addr(0x7ffe2f3038ac)
    // *b: 5, b: 0x7ffe2f3038ac, &b: 0x7ffe2f3038a0
    *a = 6;
    // *a: 6, a: 0x7ffe2f30389c, &a: 0x7ffe2f303890
    cout << "lvalue: " << lvalue << " addr(" << &lvalue << ")" << endl;
    cout << "*b: " << *b << ", b: " << b << ", &b: " << &b << endl;
    cout << "*a: " << *a << ", a: " << a << ", &a: " << &a << endl;

    cout << "---" << endl;

    // rvalue reference to the data (prvalue)
    int && rvalue = 20; // copy constructor
    // rvalue: 20 addr(0x7ffe2f30388c)

    cout << "rvalue: " << rvalue << " addr(" << &rvalue << ")" << endl;

    rvalue = 30; // copy assignment operator
    // rvalue: 30 addr(0x7ffe2f30388c)
    cout << "rvalue: " << rvalue << " addr(" << &rvalue << ")" << endl;

    // int * asd = &20; // Error
    // cout << "rvalue: " << asd << endl;
}
