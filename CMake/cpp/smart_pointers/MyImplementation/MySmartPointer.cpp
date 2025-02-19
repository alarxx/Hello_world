// https://www.geeksforgeeks.org/smart-pointers-cpp/
#include <iostream>

// --- Data ---
class Data {
private:
    int __theData;
public:
    Data(int data) : __theData(data) {
        std::cout << "Data Constructor" << "\n";
    }
    ~Data() {
        std::cout << "~Data Destructor" << "\n";
    }
    int get(){ return __theData; }
};
// ------

// --- MySmartPointer ---
template <typename T> class MySmartPointer {
private:
    T * ptr;
public:
    explicit MySmartPointer(T * ptr){
    // MySmartPointer(T * ptr){
        std::cout << "Constructor call of MySmartPointer" << std::endl;
        this->ptr = ptr;
    }
    ~MySmartPointer(){
        std::cout << "Destructor call of MySmartPointer" << std::endl;
        delete ptr;
    }
    void sayHello(){
        std::cout << "Hello, I am Smart Pointer!" << std::endl;
    }
    T& operator * (){
        return *ptr;
    }
    T * operator -> (){
        return ptr;
    }
};
// ------

void sayHello(MySmartPointer<Data> * sptr){
    std::cout << "Call by pointer" << std::endl;
    sptr->sayHello();
}
void sayHello(MySmartPointer<Data> sptr){
    std::cout << "Call by value" << std::endl;
    sptr.sayHello();
}

/**
 * Зачем нужен `explicit` конструктор?
 * Мы не хотим, чтобы происходило неявное преобразование и неявный переброс в конструктор.
 */
void error_example(){
    Data * data = new Data(123);

    // Мы не хотим, чтобы data удалялся например
    // Но, sayHello обернет указатель и удалит его после
    // sayHello(data);
    // error: could not convert ‘data’ from ‘Data*’ to ‘MySmartPointer<Data>’

    // Я хз, короче просто знай, что создание указателя должно быть только в конструкторе Smart Pointer-а.
    sayHello(MySmartPointer(new Data(123)));

    // MySmartPointer<Data> sdptr = data;
    // error: conversion from ‘Data*’ to non-scalar type ‘MySmartPointer<Data>’ requested

    // Double memory freeing
    delete data;
}


int main(){
    // MySmartPointer<Data> sdptr = data; // error: conversion from ‘Data*’ to non-scalar type ‘MySmartPointer<Data>’ requested
    MySmartPointer<Data> sdptr(new Data(123));

    sayHello(&sdptr);

    // std::cout << (*sdptr).get() << std::endl;
    std::cout << sdptr->get() << std::endl; // тут два раза вызывается arrow operator? Потому что должно быть ptr->get()

    error_example();
}
