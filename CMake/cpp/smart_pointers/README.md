
---

#### Smart Pointers. Brief

**Smart pointer** - это просто обертка над raw pointer-ом.

Types:
- `unique_ptr`
	- `std::make_unique<>()`
	- `std::move()` - отдать владение
- `shared_ptr`
	- `std::make_shared<>()`
	- `use_count()` - количество указывающих на объект
	- `reset()` - поменять объект (ControlBlock)
- `weak_ptr` - наблюдатель за значением на которое указывает `shared_ptr`
	- `lock()` - получить значение на которое указывает `shared_ptr`
	- `expired()` - из-за конкуретности лучше не использовать

 Application of `weak_ptr`:
 - Cache
 - Team one-to-many members

**Example of use**
Suppose we have class and some functions:
```cpp
#include <iostream>
#include <memory>

class Data { ... };

// By default мы должны работать вот так! call-by-value
template <typename T> void fun(std::shared_ptr<T> sptr){
    // Перед функцией создасться копия shared_ptr-а, который мы передаем
    // sptr - это копия
}

template <typename T> void fun(std::weak_ptr<T> wptr){
    // Перед функцией создасться копия weak_ptr-а, который мы передаем
    // wptr - это копия
	if(std::shared_ptr<Data> tmp = wptr.lock()){ ... }
}
```

```cpp
// Data * dataptr = new Data(123);
std::shared_ptr<Data> sptr = std::make_shared<Data>(123);
// sptr is like pointer
// *sptr - Data object

// Actually, shared_ptr holds intermediate object called ControlBlock

// Создаёт копию sptrc (call-by-value)
fun(sptr);

std::weak_ptr<Data> weak = sptr;

// Создаёт копию weakc (call-by-value)
fun(weak);

// Создается новый shared_ptr
if(auto tmp = weak.lock())
	tmp->show();
else
	std::cout << "weak is expired! \n";
```

----

####

https://www.geeksforgeeks.org/smart-pointers-cpp/

Problems with raw pointers:
- Wild Pointers
- Memory Leaks
- Dangling Pointers
- Data Inconsistency
- Buffer Overflow

Smart pointer - это просто обертка над raw pointer-ом:

```cpp
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
...

MySmartPointer<int> sptr(new int(123));
```


**Зачем нужен `explicit` конструктор?**

-> [[C++ Programming Language - explicit]]

Мы не хотим, чтобы происходило неявное преобразование и неявный переброс в конструктор.
```cpp
void fun(MySmartPointer<int> sptr){
    std::cout << "Call by value" << std::endl;
    sptr.sayHello();
}
int * num = new int(123);
// MySmartPointer<int> isptr = num; // error if explicit
fun(num); // error if explicit
delete num;
```
Мы не хотим, чтобы num удалялся например.
Но, fun обернет указатель и удалит его после.

Я хз, короче просто знай, что создание указателя должно быть только в конструкторе Smart Pointer-а.

---

- `auto_ptr` - deprecated. Проблема с `auto_ptr` была в том, что вел себя как `unique_ptr`, но при копировании передавал владение, и непонятно.
-
- `unique_ptr`
	- `std::make_unique<>()`
	- `std::move()` - отдать владение
- `shared_ptr`
	- `std::make_shared<>()`
	- `use_count()` - количество указывающих на объект
	- `reset()` - поменять объект
- `weak_ptr` - наблюдатель за значением на которое указывает `shared_ptr`
	- `lock()` - получить значение на которое указывает `shared_ptr`
	- `expired()` - из-за конкуретности лучше не использовать

---

 Application of `weak_ptr`:
 - Cache
 - Team one-to-many members

---

https://stackoverflow.com/questions/12030650/when-is-stdweak-ptr-useful
https://stackoverflow.com/questions/5671241/how-does-weak-ptr-work

**Как отследить, что `shared_ptr` удалился**
```cpp
#include <iostream>
#include <memory>

/*
// OLD, problem with dangling pointer
// PROBLEM: ref will point to undefined data!
int* ptr = new int(10);
int* ref = ptr;
delete ptr;
*/

std::shared_ptr<Data> sptr = std::make_shared<Data>(123);
// *sptr - Data object

// Как отследить, что sptr удалился?
// Мы не можем просто создать другой shared_ptr, чтобы отслеживать sptr
// std::shared_ptr<Data> sptr2 = sptr; // No
// Circular Dependency of shared pointers never deallocates what leads to Memory Leak
std::weak_ptr<Data> weak1 = sptr;

ptr1.reset(new Data(321));
// Теперь он ссылается на новый ControlBlock
// И соответсвенно уменьшил владельцев старого ControlBlock-а
// Но так же обычно не делают, меняют значение на которое указывает указатель, а не полностью объект

std::weak_ptr<Data> weak2 = sptr;

// weak1 is expired!
// expired() лучше не использовать и предпочесть lock() из-за race condition
// weak1 все еще ссылается на старый ControlBlock
// старый ControlBlock не удаляется, так как все ещё имеет наблюдающих-weak_ptr
// Не удаляем, чтобы оставшиеся weak_ptr не указывали на мусор
if(auto tmp = weak1.lock()) {
	// tmp создается из старого ControlBlock
	// tmp's controlBlock.object is nullptr, поэтому bool(tmp) false
	tmp->show(); // (*tmp).show() i.e. Data.show()
}
else {
	std::cout << "weak1 is expired! \n"; // <--
}

// weak2 points to new Foo
if(auto tmp = weak2.lock())
	tmp->show(); // <--
else
	std::cout << "weak2 is expired! \n";
```

Почему когда мы делаем reset наш `weak_ptr` становится expired? Как будто `weak_ptr` указывает на сам сокрытый объект shared pointer-а

**Control Block**

https://en.cppreference.com/w/cpp/memory/shared_ptr

В общем, `shared_ptr` и `weak_ptr` не указывают на сам объект, а указывают на **control block** (intermediate object) which wrap the object itself and holds:
- managed object
- deleter
- allocator
- number of `shared_ptr` that own the object
- number of `weak_ptr` that own the object

>Destructor of `shared_ptr` decrements the number of shared owners of the control block, and if that counter reaches zero, the control block calls the destructor of the managed object. Control block does not deallocate itself until the `weak_ptr` counter reaches zero.

Когда удаляет `shared_ptr`, его деструктор вызывает функцию декремент `control block`-а.
Функция декрменет `control block`-а уменьшает number of `shared_ptr`, и если это число 0, то она вызывает деструктор самого объекта, на который мы указывали.
Но, `control block` не удаляется пока число `weak_ptr` не станет 0, потому что иначе `weak_ptr` были бы висячими указателями, а адрес памяти мог бы уже быть выделен другой задачей.

---

#### My Implementation of Smart Pointers and Control Block

```cpp
// --- ControlBlock ---
template <typename T>
class ControlBlock {
private:
    T * object;
    int shared_number, weak_number;
public:
    ControlBlock(T * object){
        this->object = object;
        shared_number = 0;
        weak_number = 0;
    }
    ~ControlBlock(){}
    T * get(){ return object; }
    void increment_shared(){ shared_number++; }
    void increment_weak(){ weak_number++; }
    void decrement_shared(){
		if(shared_number == 0)
			return; // need it on weak.lock()
		shared_number--;
        if(shared_number == 0){
        // Выполняется только когда значение становится 0
            delete object;
            object = nullptr;
            if(weak_number == 0){
                delete this;
            }
        }
    }
    void decrement_weak(){
        weak_number--;
        if(weak_number == 0 && shared_number <= 0){
            delete this;
        }
    }
};
// ------

// --- MySmartPointer ---
// Заставляет ControlBlock удерживать объект
template <typename T>
class MySmartPointer {
private:
    ControlBlock<T> * controlBlock; // Stack
public:
	explicit MySmartPointer(T * newobj){
		controlBlock = new ControlBlock<T>(newobj);
        controlBlock->increment_shared();
	}
	// A a = b; // We can use like that
	MySmartPointer(const MySmartPointer<T> & other){...}
	explicit MySmartPointer(ControlBlock<T> * controlBlock){...}
	MySmartPointer<T> & operator = (const MySmartPointer<T> & other) = delete;
	~MySmartPointer(){...}
	explicit operator bool() const {
        return controlBlock && controlBlock->get() != nullptr;
    }
...
}
// ------

// --- MyWeakPointer ---
// Не заставляет ControlBlock удерживать объект
// Но все еще удерживает ControlBlock
template <typename T>
class MyWeakPointer {
private:
    ControlBlock<T> * controlBlock;
public:
    MyWeakPointer(const MyWeakPointer<T> & other){...}
    MyWeakPointer(const MySmartPointer<T> & mySmartPointer){...}
    MyWeakPointer() = delete;
	// Мы никогда не используем Assign Operator
	MyWeakPointer<T> & operator = (const MySmartPointer<T> & mySmartPointer) = delete;
	MyWeakPointer<T> & operator = (const MyWeakPointer<T> & other) = delete;
	~MyWeakPointer(){...}
    MySmartPointer<T> lock(){
        MySmartPointer<Data> sptr(controlBlock); // in Stack
        return sptr; // -> Named Return Value Optimization
    }
...
}
// ------
```

```cpp
MySmartPointer<Data> sptr(new Data(123));
// sptr is like pointer
// *sptr - Data object

MySmartPointer<Data> sptr2 = sptr; // MySmartPointer<Data> sptr2(sptr);

MyWeakPointer<Data> weak = sptr;
if(MySmartPointer<Data> tmp = weak.lock()){
	// *tmp
}
```

---


**Как работать с функциями?**
Если мы используем Smart Pointer-ы, то нам всегда стоит использовать call-by-value.
Но, при этом добавляется небольшой overhead на подсчеты владельцев и копирование этих Smart Pointer-ов.

**Происходит ли копирование при call-by-value?**
Да, происходит.

**А если вернуть из функции объект allocated in Stack?**
Объкт in Stack не удалится, он перейдет в место откуда вызывали функцию, и уже там будет жить.
-> Named Return Value Optimization
