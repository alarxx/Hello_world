# lvalue, rvalue and move semantics

[[C++ Programming Language]]

## lvalue = rvalue

**References:**
- https://www.geeksforgeeks.org/lvalues-references-and-rvalues-references-in-c-with-examples/
- https://stackoverflow.com/questions/3601602/what-are-rvalues-lvalues-xvalues-glvalues-and-prvalues

![[Pasted image 20250220011042.png]]
- **lvalue**
	- glvalue (generalized value)
	- xvalue (expiring value)
- **rvalue**
	- prvalue (pure value)

So-called historically because of their appearances on the sides of expressions:
`lvalue = rvalue`
- lvalue - have a name and address (обычная переменная)
- rvalue - temporary value that disappear after the expression they appear in
- lvalue and rvalue references

```cpp
int b { 10 };
int & lvalue = b; // regular reference to a memory location
int && rvalue = 20; // rvalue reference to the data (prvalue)
```

Литерал 20 здесь это pure rvalue.
У литералов (prvalue) нет адреса (&20 -> Error).
rvalue не может принимать ничего кроме prvalue.
При создании rvalue reference создается ячейка в которой будет хранится prvalue.
int && не указатель на адрес, это переменная-ссылка, которая принимает только prvalue.

---

**lvalue**
```cpp
// --- Copy ---
int _b = 10;
int * b = &_b;
// _b: 10 addr(0x7ffe2f3038ac)
// *b: 10, b: 0x7ffe2f3038ac, &b: 0x7ffe2f3038a0

// regular reference variable
int & lvalue = *b; // copy constructor
// lvalue: 10 addr(0x7ffe2f3038ac)

// --- Assignment ---
int _a = 5;
int * a = &_a;
// _a: 5 addr(0x7ffe2f30389c)
// *a: 5, a: 0x7ffe2f30389c, &a: 0x7ffe2f303890

lvalue = *a; // copy assignment operator
// lvalue: 5 addr(0x7ffe2f3038ac)
// *b: 5, b: 0x7ffe2f3038ac, &b: 0x7ffe2f3038a0
*a = 6;
// *a: 6, a: 0x7ffe2f30389c, &a: 0x7ffe2f303890
```
Здесь заметь разницу между copy constructor-ом и copy assignment operator-ом. Когда мы создали `lvalue` она стала ссылкой `_b`, а когда мы за-assign-или `_a` то мы поменяли `_b` этим значением.

**rvalue**
```cpp
// rvalue reference to the data (prvalue)
int && rvalue = 20; // copy constructor
// rvalue: 20 addr(0x7ffe2f30388c)

cout << "rvalue: " << rvalue << " addr(" << &rvalue << ")" << endl;

rvalue = 30; // copy assignment operator
// rvalue: 30 addr(0x7ffe2f30388c)
cout << "rvalue: " << rvalue << " addr(" << &rvalue << ")" << endl;

// int * asd = &20; // Error
// cout << "rvalue: " << asd << endl;
```
Здесь то же самое, сперва мы создаем, а потом меняем само значение.

---

## Functions

**References:**
- https://medium.com/@weidagang/demystifying-std-move-in-c-c4f43559995f
- https://stackoverflow.com/questions/3413470/what-is-stdmove-and-when-should-it-be-used

```cpp
// call-by-reference
void fun(int & arg){
    cout << "&arg: " << arg << " addr(" << &arg << ")" << endl;
}

// fun принимает только временные объекты (rvalue) и привязывает к `arg`, при этом создавая для `arg` memory address
void fun(int && arg){
    // a - lvalue уже, но мы передаем сюда rvalue
    cout << "&&arg: " << arg << " addr(" << &arg << ")" << endl;
}
```

```cpp
int a = 15; // a - lvalue
```

```cpp
fun(a); // call-by-reference
```

```cpp
fun(10); // pure rvalue

// explicitly создаем копию - rvalue
fun(int(a));
```

**Move semantics:**
```cpp
fun(std::move(a)); // "относись как к rvalue"
// but works like call-by-reference
// `a` is still available here
```
imo, здесь как-будто обычный call-by-reference идет, просто в таком случае можно реализовать функцию подругому.

### Move semantics

Какая операция происходит когда мы передаем в функцию переменную через `std::move`?
Выглядит будто это обычное call-by-reference, ничего не крадется, переменная все еще доступна позже.
==`std::move` кастит в `rvalue`, а не реально перемещает.==
-> [[C++ Programming Language - lvalue, rvalue and move semantics#`std move`]]

По идее, кроме move конструктора и move assignment operator-а, ==нет никакой причины, чтобы использовать `rvalue` в виде аргумента в функциях==, потому что по умолчанию это работает как обычный call-by-reference.
-> [[C++ Programming Language - The rule of 3-5-0]]

Можно конечно так намекнуть, что аргумент будет полностью "украден" этой функцией, и внутри реализовать это, но это не по [[Functional Programming (FP)|FP]] (с pure function мы не можем менять состояние аргумента):
```cpp
void fun(Data && arg){
	Data stealed = std::move(arg);
	...
}
...
fun(std::move(data));
// data is empty now
```

Единственное реально нужное применение, это когда мы создаем или назначаем объект с `std::move`:
```cpp
Data data1(100);
Data data2(200)
// data1 holds 100
// data2 holds 200

Data stealed = std::move(data1);
// stealed holds 100
// data1 holds -1 (nullptr)

Data copy(-1);
copy = data2;
// copy holds 200
// data2 holds -1 (nullptr)
```

---

Функции принимающие && rvalue работают абсолютно идентично, как call-by-reference &, не вызываются move constructor-ы и соответственно `a` не удаляется.

Оказывается, чтобы при передаче в функцию вызывался move constructor нужно чтобы функция принимала call-by-value.

При передаче аргументов в call-by-value вызываются copy или move конструкторы в зависимости от lvalue или rvalue.
Call-by-value может принимать и lvalue, и rvalue, что создает ambiguous проблему:
```
error: call of overloaded ‘fun(Data&)’ is ambiguous
	fun(a);
note: candidate: ‘void fun(Data)’
note: candidate: ‘void fun(Data&)’
```

Call-by-value всегда вызывает copy/move constructor:
```cpp
void fun(Data a){ ... }

fun(std::move(a)); // "относись как к rvalue"
// calls move constructor
// a is empty now
```

Я это понял, когда попробовал поиграться с `unique_ptr`:
```cpp
#include <iostream>
#include <memory>

void takeOwnership(std::unique_ptr<int> ptr){
	std::cout << "function owns: " << *ptr << std::endl;
}

int main(){
	std::unique_ptr<int> uptr = std::make_unique<int>(100);
    std::cout << "uptr: " << *uptr << std::endl;
	takeOwnership(std::move(uptr));
	if(!uptr){
		std::cout << "uptr: " << *uptr << std::endl;
	}
}
```
-> [[C++ Programming Language - Smart Pointers]]

---

**Push to vector example**
https://www.geeksforgeeks.org/stdmove-in-utility-in-c-move-semantics-move-constructors-and-move-assignment-operators/
```cpp
std::vector<std::string> vec;
vec.reserve(3);

// constructing & initializing
// a string with "Hello"
std::string str("Hello");

// Inserting a copy of string
// object
vec.push_back(str); // lvalue

// Inserting a copy of an
// temporary string object
vec.push_back(str + str); // rvalue

// Again inserting a copy of
// string object
vec.push_back(std::move(str)); // it may not create a copy
// now str is empty
```

---

## `std::move` implementation

**References:**
- https://stackoverflow.com/questions/7510182/how-does-stdmove-convert-expressions-to-rvalues
- https://en.cppreference.com/w/cpp/types/remove_reference

`std::move` - это функция, которая приводит объект в rvalue ссылке (T&&):
```cpp
template <typename T>
typename remove_reference<T>::type&& move(T&& arg){
	return static_cast<typename remove_reference<T>::type&&>(arg);
}
```

https://en.cppreference.com/w/cpp/types/remove_reference
remove_reference используется для получения чистого типа
int& -> int
int&& -> int

**rvalue**
```cpp
Object a = std::move(Object());
// Object() is temporary (no name), which is prvalue
```

 `move` with `[T = Object]`:
```cpp
remove_reference<Object>::type&& move(Object&& arg){
  return static_cast<remove_reference<Object>::type&&>(arg);
}
// So we get ->
Object&& move(Object&& arg){
  return static_cast<Object&&>(arg);
}
```

**lvalue**
```cpp
Object a; // a is lvalue
Object b = std::move(a);
```

`move` with `[T = Object&]`:
```cpp
remove_reference<Object&>::type&& move(Object& && arg){
  return static_cast<remove_reference<Object&>::type&&>(arg);
}
// So we get ->
Object&& move(Object& && arg){
  return static_cast<Object&&>(arg);
}
/*
What does `Object& &&` mean?
Object &  &  = Object &
Object &  && = Object &
Object && &  = Object &
Object && && = Object &&
*/
// So we get ->
Object&& move(Object& arg){
  return static_cast<Object&&>(arg);
}
```

---

## Notes

- `std::boolalpha` вывод будет false/true, без него вывод 0/1.
- `noexcept`

---

# The rule of 3/5/0

<- [[C++ Programming Language - lvalue, rvalue and move semantics]]

---

https://en.cppreference.com/w/cpp/language/rule_of_three

- destructor
- copy constuctor
- copy assignment operator
- move constructor
- move assignment operator

Когда мы реализуем деструктор, нам скорее всего нужно реализовать copy constructor и copy assignment operator.
Если мы точно не используем копирование, то их можно либо сделать `private`, либо `= delete;`.

---

## Rule of zero

> [If you can avoid defining default operations, do](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#Rc-zero)

Если ты не определяешь деструктор, то стоит оставить default реализацию и не определять copy/move конструкторы и copy/move assignment operator-ы.

Но, если нам нужно создать `virtual` destructor, то можно указать `default`:
```cpp
class A {
public:
	// Copy
	A(const A & other) = default;
	A & operator = (const A & other) = default;
	// Move
	A(A && other) = default;
	A & operator = (A && other) = default;
	// Destructor
	virtual ~A() = default;
}
```

Как работает move semantics при default реализациях?
На сколько я понимаю он просто создает копии всех полей класса.
И это ок, если мы не выделяем память динамически.
То есть если скопируется указатель, то 2 указателя будут указывать на один и тот же адрес, и проблема в том, что когда 1 удалится, то другой будет указывать на освобожденный адрес, и когда уже другой удалится, то будет core dump.

---

## Example (5)

Понятно как работает copy constructor и copy operator,
==как и зачем нужен move== соответствующий

```cpp
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
        cout << "Copy Constructor" << endl;
        __theData = new int(*other.__theData);
    }
    // Data(Data && other) = default;
    Data(Data && other){
        cout << "Move Constructor" << endl;
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
    // Data & operator = (Data & other) = default;
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

// fun принимает только временные объекты (rvalue) и привязывает к a, при этом создавая для a memory address
void fun(Data && a){
    // a - lvalue уже, но мы передаем сюда rvalue
    cout << "&&arg: " << a.get() << " addr(" << &a << ")" << endl;
}
void fun(Data & a){
    // a - lvalue уже, но мы передаем сюда rvalue
    cout << "&arg: " << a.get() << " addr(" << &a << ")" << endl;
}
// ------

int main(){
    Data a(15); // a - lvalue
    cout << "a: " << a.get() << " addr(" << &a << ")" << endl;

    fun(a); // call-by-reference

    fun(Data(10));

    fun(std::move(a)); // "относись как к rvalue"
    cout << "a: " << a.get() << " addr(" << &a << ")" << endl;

    // Copy Constructor
    Data a_copy = a;
    cout << "a: " << a.get() << " addr(" << &a << ")" << endl;
    // Move Constructor
    Data a_move = std::move(a);
    // a is empty
    cout << "a: " << a.get() << " addr(" << &a << ")" << endl;
}
```
