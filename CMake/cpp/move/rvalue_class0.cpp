/*

Просто хотел проверить как работает move semantics при default реализациях

Как работает move semantics при default реализациях?
На сколько я понимаю он просто создает копии всех полей класса.
И это ок, если мы не выделяем память динамически.
То есть если скопируется указатель, то 2 указателя будут указывать на один и тот же адрес, и проблема в том, что когда 1 удалится, то другой будет указывать на освобожденный адрес, и когда уже другой удалится, то будет core dump.

 */

#include <iostream>

using std::cout, std::endl;

// --- Data ---
class Data {
private:
    int __theData;
public:
    Data() = delete;
    Data(const int data){
        __theData = data;
        cout << "Constructor" << endl;
    }
    // int * getptr(){ return __theData; }
    int get(){
        return __theData;
    }

    virtual ~Data() = default; // без virtual, можно было бы вообще не писать эту строку
    Data(const Data & other) = default;
    Data(Data && other) = default;
    Data & operator = (const Data & other) = default;
    Data & operator = (Data && other) = default;
};
// ------

void fun(Data a){
    // a - lvalue уже, но мы передаем сюда rvalue
    cout << "arg: " << a.get() << " addr(" << &a << ")" << endl;
}

int main(){
    Data a(15); // a - lvalue
    cout << "a: " << a.get() << " addr(" << &a << ")" << endl;
    cout << "1------" << endl;

    fun(a); // call-by-reference
    cout << "2------" << endl;

    fun(Data(10));
    cout << "3------" << endl;

    fun(std::move(a)); // "относись как к rvalue"
    cout << "a: " << a.get() << " addr(" << &a << ")" << endl;
    cout << "4------" << endl;

    // Copy Constructor
    Data a_copy = a;
    cout << "a: " << a.get() << " addr(" << &a << ")" << endl;
    cout << "a_copy: " << a_copy.get() << " addr(" << &a_copy << ")" << endl;
    cout << "5------" << endl;

    // Move Constructor
    Data a_move = std::move(a);
    // a is empty
    cout << "a: " << a.get() << " addr(" << &a << ")" << endl;
    cout << "a_move: " << a_move.get() << " addr(" << &a_move << ")" << endl;
}

