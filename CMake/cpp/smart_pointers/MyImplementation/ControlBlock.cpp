/*
    SPDX-License-Identifier: MPL-2.0
    --------------------------------
    This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
    If a copy of the MPL was not distributed with this file,
    You can obtain one at https://mozilla.org/MPL/2.0/.

    Provided “as is”, without warranty of any kind.

    Copyright © 2025 Alar Akilbekov. All rights reserved.

 * References:
 * https://www.geeksforgeeks.org/smart-pointers-cpp/
 * https://stackoverflow.com/questions/12030650/when-is-stdweak-ptr-useful
 * https://stackoverflow.com/questions/5671241/how-does-weak-ptr-work
 * https://en.cppreference.com/w/cpp/memory/shared_ptr
 */


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

// --- ControlBlock ---

template <typename T> class ControlBlock {
private:
    T * object;
    int shared_number, weak_number;
public:
    ControlBlock(T * object){
        std::cout << "ControlBlock Constructor" << "\n";
        this->object = object;
        shared_number = 0;
        weak_number = 0;
    }
    ~ControlBlock(){
        std::cout << "~ControlBlock Destructor" << "\n";
    }
    T * get(){
        return object;
    }
    int use_count(){
        return shared_number;
    }
    void printCount(){
        std::cout << "printCount:" << "\n";
        std::cout << "\tshared_number : " << shared_number << "\n";
        std::cout << "\tweak_number   : " << weak_number << "\n";
    }
    void increment_shared(){
        shared_number++;
        std::cout << "increment_shared, incremented to: " << shared_number << "\n";
    }
    void increment_weak(){
        weak_number++;
        std::cout << "weak_number, incremented to: " << weak_number << "\n";
    }
    void decrement_shared(){
        // if(shared_number == 0){ // need it on weak.lock()
        //     return;
        // }
        shared_number--; // Без if0 может быть меньше 0
        // Тут короче все дело в выводе красивом, мы хотим уменьшить потом вывести
        std::cout << "decrement_shared, decremented to: " << shared_number << "\n";
        if(shared_number == 0){ // Выполняется только когда значение становится 0
            delete object;
            object = nullptr;
            if(weak_number == 0){
                delete this;
            }
        }
    }
    void decrement_weak(){
        weak_number--;
        std::cout << "decrement_weak, decremented to: " << weak_number << "\n";
        if(weak_number == 0 && shared_number <= 0){
            delete this;
        }
    }
};

// ------


// --- MySmartPointer ---

template <typename T> class MySmartPointer {
private:
    ControlBlock<T> * controlBlock; // Stack
public:
    explicit MySmartPointer(T * newobj){
    // MySmartPointer(T * ptr){
        std::cout << "{---start-Constructor call of MySmartPointer(T)---" << "\n";
        // pointer is in Stack, but created object is in Heap
        controlBlock = new ControlBlock<T>(newobj);
        controlBlock->increment_shared();
        std::cout << "}---end---Constructor call of MySmartPointer(T)---" << "\n\n";
    }
    void reset(T * newobj){
        // from Destructor
        controlBlock->decrement_shared();
        // from Constructor
        controlBlock = new ControlBlock<T>(newobj);
        controlBlock->increment_shared();
    }
    explicit MySmartPointer(ControlBlock<T> * controlBlock){
        std::cout << "{---start-Constructor call of MySmartPointer(ControlBlock)---" << "\n";

        // То есть здесь когда мы передаем по указателю, сам указатель копируется, но указывает на тот же самый адрес controlBlock
        std::cout << "controlBlock &pointer->address: " << &controlBlock << "->" << controlBlock << "\n";
        this->controlBlock = controlBlock; // Указатель на адрес ControlBlock-а, то есть &(*controlBlock)
        if(controlBlock->use_count() == 0){
            std::cout << "Control block is empty!\n";
        }
        else {
            this->controlBlock->increment_shared();
        }
        std::cout << "}---end---Constructor call of MySmartPointer(ControlBlock)---" << "\n\n";
    }
    ~MySmartPointer(){
        std::cout << "{~~~start~Destructor call of MySmartPointer~~~" << "\n";
        controlBlock->decrement_shared(); // ControlBlock должен сам себя удалить, если никто не ссылается на него
        std::cout << "}~~~end~~~Destructor call of MySmartPointer~~~" << "\n\n";
    }

    /** Копирование - Передача владения через =
     *
     * Присваивание вызывает конструктор копирования?
     * Когда создаётся новый объект и инициализируется существующим объектом того же типа, используется конструктор копирования, если он доступен.
     * Здесь происходит Copy Constructor, а не Assignment, потому что sptr2 previously wasn't instanciated:

        MySmartPointer<Data> sptr2 = sptr;

     */
    MySmartPointer(const MySmartPointer<T> & other){
    // explicit MySmartPointer(const MySmartPointer<T> & other){
        std::cout << "{---start-Copy Constructor call of MySmartPointer(MySmartPointer)---" << "\n";
        if(this == &other){ // IMO это никогда не выполнится: A a(a);
            throw "What the hell!";
        }
        this->controlBlock = other.controlBlock;
        this->controlBlock->increment_shared();
        std::cout << "}---end---Copy Constructor call of MySmartPointer(MySmartPointer)---" << "\n\n";
    }

    /** Copy Opeartor or Assignment Operator =
     * Выполняется если сначала создаем объект, потом assign-им другой объект
      A b;
      A a;
      a = b;
     */
    MySmartPointer<T> & operator = (const MySmartPointer<T> & other) = delete;

    void sayHello(){
        std::cout << "Hello, I am Smart Pointer!" << "\n";
    }
    T& operator * (){
        return *(controlBlock->get());
    }
    T * operator -> (){
        return controlBlock->get();
    }

    ControlBlock<T> * getControlBlock() const {
        return controlBlock;
    }

    explicit operator bool() const {
        // weak_ptr не считаются
        return controlBlock && controlBlock->get() != nullptr;
    }

    int use_count(){
        return controlBlock->use_count();
    }
    void printCount(){
        controlBlock->printCount();
    }

};

// ------

// --- MyWeakPointer ---

template <typename T> class MyWeakPointer {
private:
    ControlBlock<T> * controlBlock;
public:
    MyWeakPointer() = delete;

    // Copy constructor
    // Calls on call-by-value or creating weak_ptr
    // MyWeakPointer(const MyWeakPointer<T> & other) = delete;
    MyWeakPointer(const MyWeakPointer<T> & other){
        std::cout << "{---start-Copy Constructor call of MyWeakPointer(MyWeakPointer)---" << "\n";
        if(this == &other) { // IMO это никогда не выполнится: A a(a);
            throw "What the hell!";
        }
        controlBlock = other.controlBlock;
        controlBlock->increment_weak();
        std::cout << "}---end---Copy Constructor call of MyWeakPointer(MyWeakPointer)---" << "\n\n";
    }

    MyWeakPointer(const MySmartPointer<T> & mySmartPointer){
        std::cout << "{---start-Constructor call of MyWeakPointer(MySmartPointer)---" << "\n";
        controlBlock = mySmartPointer.getControlBlock();
        controlBlock->increment_weak();
        std::cout << "}---end---Constructor call of MyWeakPointer(MySmartPointer)---" << "\n\n";
    }

    /*
    MyWeakPointer<T> & operator = (const MySmartPointer<T> & mySmartPointer){
        std::cout << "Assignment Operator call of MyWeakPointer(MySmartPointer)" << "\n";
        this->controlBlock->decrement_weak();
        this->controlBlock = mySmartPointer.getControlBlock();
        this->controlBlock->increment_weak();
        return *this;
    }
    */
    MyWeakPointer<T> & operator = (const MySmartPointer<T> & mySmartPointer) = delete;

    MyWeakPointer<T> & operator = (const MyWeakPointer<T> & other) = delete;

    ~MyWeakPointer(){
        std::cout << "{~~~start~Destructor call of MyWeakPointer~~~" << "\n";
        controlBlock->decrement_weak();
        std::cout << "}~~~end~~~Destructor call of MyWeakPointer~~~" << "\n\n";
    }

    /**
     * Когда нужно получить объект и быть уверенным, что он не удалится (зачукать)
     */
    MySmartPointer<T> lock(){
        std::cout << "{---start-lock()---" << "\n";
        std::cout << "controlBlock &pointer->address: " << &controlBlock << "->" << controlBlock << "\n";
        // То есть здесь когда мы передаем по указателю, сам указатель копируется, но указывает на тот же самый адрес controlBlock
        MySmartPointer<Data> sptr(controlBlock); // in Stack, but ... it's a pointer, strange
        // По идее должна создаться еще одна копия, а конкретно этот sptr должен удалиться.
        // -> Named Return Value Optimization
        std::cout << "}---end-lock()---" << "\n\n";
        return sptr;
    }
};

// ------


template <typename T> void fun(MyWeakPointer<T> wptr){
    // Перед функцией создасться копия weak_ptr-а, который мы передаем
    // wptr - это копия
    std::cout << "---start-funny fun function---" << "\n";
    if(MySmartPointer<Data> tmp = wptr.lock()){
        tmp.printCount();
    }
    else {
        std::cout << "wptr is expired!" << std::endl;
    }
    std::cout << "---end--funny fun function---" << "\n\n";
}


// By default мы должны работать вот так! call-by-value
template <typename T> void sayHello(MySmartPointer<T> sptr){
    // Перед функцией создасться копия shared_ptr-а, который мы передаем
    // sptr - это копия
    std::cout << "Call by value" << std::endl;
    sptr.printCount();
}
// Мы никогда не должны создавать pointer to SmartPointer.
template <typename T> void sayHello(MySmartPointer<T> * sptr){
    std::cout << "Call by pointer" << std::endl;
    sptr->printCount();
}


int main(){
    // // ControlBlock<Data> c(new Data(123)); // Мы не должны так использовать ControlBlock
    // ControlBlock<Data> * c = new ControlBlock<Data>(new Data(123));
    // c->increment_shared();
    //
    // c->printCount();
    // std::cout << (c->get())->get() << std::endl;
    //
    // c->decrement_shared(); // Теперь ControlBlock должен удалиться

    std::cout << "\n" << "1. " << std::endl;

    MySmartPointer<Data> sptr(new Data(123));
    // *sptr - Data object

    // Создаёт копию sptrc (call-by-value)
    sayHello(sptr);
    // Удалит копию sptrc (call-by-value)

    std::cout << "\n" << "2. " << std::endl;
    {
        std::cout << "{}" << std::endl;
        // https://en.cppreference.com/w/cpp/language/operators
        // https://stackoverflow.com/questions/11706040/whats-the-difference-between-assignment-operator-and-copy-constructor
        // Здесь происходит Copy Constructor, а не Assignment, потому что sptr2 previously wasn't instanciated
        // Почему присваивание вызывает конструктор копирования?
        // Когда создаётся новый объект и инициализируется существующим объектом того же типа, используется конструктор копирования, если он доступен.
        MySmartPointer<Data> sptr2 = sptr;
        sptr2.printCount();
    } // sptr2 deleted

    std::cout << "\n" << "3. " << std::endl;
    // Как отследить, что sptr удалился?
    // Мы не можем просто создать другой shared_ptr, чтобы отслеживать sptr
    // std::shared_ptr<Data> sptr2 = sptr; // No
    // Circular Dependency of shared pointers never deallocates what leads to Memory Leak
    MyWeakPointer<Data> weak1 = sptr;
    // При передаче создасться копия weak_ptr
    // внутри мы будем вызывать lock() -> shared_ptr
    fun(weak1);
    // 1 spdr
    // 1 weak1
    // 2 copy weak1 (call-by-value)
    // 2 lock() -> spdr2

    std::cout << "\n" << "4. " << std::endl;
    sptr.reset(new Data(321));
    // Теперь он ссылается на новый ControlBlock
    // И соответсвенно уменьшил владельцев старого ControlBlock-а
    sptr.printCount();

    // weak1 is expired!
    // expired() лучше не использовать и предпочесть lock() из-за race condition
    // weak1 все еще ссылается на старый ControlBlock
    // старый ControlBlock не удаляется, так как все ещё имеет наблюдающих-weak_ptr
    // Не удаляем, чтобы оставшиеся weak_ptr не указывали на мусор
    if(MySmartPointer<Data> tmp = weak1.lock()){
        // tmp создается из старого ControlBlock
        // tmp's controlBlock.object is nullptr, поэтому bool(tmp) false
        tmp.printCount();
    }
    else {
        std::cout << "weak1 is expired!" << std::endl;
    }

    /*
     Как работать с функциями?
     Если мы используем Smart Pointer-ы, то нам всегда стоит использовать call-by-value.
     Но, при этом добавляется небольшой overhead на подсчеты владельцев и копирование этих Smart Pointer-ов.

     Происходит ли копирование при call-by-value?
     Да, происходит.

     А если вернуть из функции объект allocated in Stack?
     Объкт in Stack не удалится, он перейдет в место откуда вызывали функцию, и уже там будет жить.
     -> Named Return Value Optimization
     */
}
