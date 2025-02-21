
#include <iostream>

using std::cout, std::endl;

// --- Data ---
class Data {
private:
    int * __theData;
public:
    Data() = delete;
    Data(const int data){
        __theData = new int(data);
        cout << "Constructor" << endl;
    }
    // int * getptr(){ return __theData; }
    int get(){
        return (__theData != nullptr) ? *__theData : -1;
    }

    // virtual ~Data() = default;
    virtual ~Data(){
        if(__theData != nullptr){
            delete __theData;
        }
    }

    // Data(const Data & other) = default;
    Data(const Data & other){
        cout << "Copy Constructor" << (other.__theData == nullptr ? " (nullptr)" : "") << endl;
        if(other.__theData != nullptr){
            __theData = new int(*other.__theData);
        }
        else {
            __theData = nullptr;
        }
    }
    // Data(Data && other) = default;
    Data(Data && other){
        cout << "Move Constructor" << (other.__theData == nullptr ? " (nullptr)" : "") << endl;
        __theData = other.__theData;
        other.__theData = nullptr;
    }
    // Почему здесь не используется const ?
    // Потому что мы должны освободить другую сторону при move

    // Data & operator = (const Data & other) = default;
    Data & operator = (const Data & other){
        cout << "Copy Assignment Operator" << endl;
        if(this != &other){
            if(__theData != nullptr){
                delete __theData;
            }
            __theData = new int(*other.__theData);
            // Либо можно просто поменять само значение
            // *this.__theData = *other.__theData;
        }
        return *this;
    }
    // Data & operator = (Data && other) = default;
    Data & operator = (Data && other){
        cout << "Move Assignment Operator" << endl;
        if(this != &other){
            __theData = other.__theData;
            if(other.__theData != nullptr){
                other.__theData = nullptr;
            }
        }
        return *this;
    }
};
// ------

// --- functions ---
/*
fun принимает только временные объекты (rvalue) и привязывает к a, при этом создавая для a memory address
void fun(Data && a){
    // a - lvalue уже, но мы передаем сюда rvalue
    cout << "&&arg: " << a.get() << " addr(" << &a << ")" << endl;
}
void fun(Data & a){
    // a - lvalue уже, но мы передаем сюда rvalue
    cout << "&arg: " << a.get() << " addr(" << &a << ")" << endl;
}
*/
void fun(Data a){
    // a - lvalue уже, но мы передаем сюда rvalue
    cout << "arg: " << a.get() << " addr(" << &a << ")" << endl;
}
/*
 Я здесь изначально использовал функции принимающие & и &&. Но они работают абсолютно идентично, как call-by-reference,
 не вызываются move constructor-ы и соответственно `a` не удаляется.
 Оказывается, чтобы при передаче в функцию вызывался move constructor нужно чтобы функция принимала call-by-value.
 При передаче аргументов в call-by-value вызываются copy или move конструкторы в зависимости от lvalue или rvalue.
 Call-by-value может принимать и lvalue, и rvalue, что создает ambiguous проблему:
    error: call of overloaded ‘fun(Data&)’ is ambiguous
        fun(a);
    note: candidate: ‘void fun(Data)’
    note: candidate: ‘void fun(Data&)’

 Call-by-value всегда вызывает copy/move constructor:

    void fun(Data a){ ... }

    fun(std::move(a)); // "относись как к rvalue"
    // calls move constructor
    // a is empty now

 */
// ------

int main(){
    Data a(15); // a - lvalue
    cout << "a: " << a.get() << " addr(" << &a << ")" << endl;
    cout << "------" << endl;

    fun(a); // call-by-reference
    cout << "------" << endl;

    fun(Data(10));
    cout << "------" << endl;

    fun(std::move(a)); // "относись как к rvalue"
    cout << "a: " << a.get() << " addr(" << &a << ")" << endl;
    cout << "------" << endl;

    // Copy Constructor
    Data a_copy = a;
    cout << "a: " << a.get() << " addr(" << &a << ")" << endl;
    cout << "a_copy: " << a_copy.get() << " addr(" << &a_copy << ")" << endl;
    cout << "------" << endl;

    // Move Constructor
    Data a_move = std::move(a);
    // a is empty
    cout << "a: " << a.get() << " addr(" << &a << ")" << endl;
    cout << "a_move: " << a_move.get() << " addr(" << &a_move << ")" << endl;
}

