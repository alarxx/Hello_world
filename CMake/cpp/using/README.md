# `typedef` and `using`
---

- https://stackoverflow.com/questions/10747810/what-is-the-difference-between-typedef-and-using
- https://www.geeksforgeeks.org/cpp-using-vstypedef/

`typedef` и `using` используются для создания псевдонимов типам (aliases).
Aliases можно сделать только для типов, но не переменных или объектов.

`using` используется для `namespace`-ов и `template`-ов.

---

```cpp
#include <iostream>

// using используется для namespace-ов и template-ов
using std::cout, std::endl;
// using cout = std::cout; // Error
// Можно еще использовать препроцессор
#define out std::cout
```

---

**Basic Type Aliases**

Насколько я понимаю typedef почти то же самое, что и using.

```cpp
// typedef int MyInt;
using MyInt = int;
MyInt a = 123;

// typedef int IntArray[5];
using IntArray = int[5];
IntArray arr; // arr[i]
```

---

**Function Pointer Type**
```cpp
// Function to point from function aliases
void fun(int a){}
```

```cpp
// Указатель на функцию, которая принимает int и возвращает void
using FunPtr = void(*)(int);
// void (*funptr)(int) = fun;
FunPtr funptr = fun;
funptr(10);
```

---

```cpp
void f1(int a){ cout << "f1: " << a << endl; }
void f2(int a){ cout << "f2: " << a << endl; }
void f3(int a){ cout << "f3: " << a << endl; }
```

```cpp
// массив указателей на функции возвращающие void
using ActionArray = void(*[3])(int);

// void(* arrs[3])(int args) = {f1, f2, f3};
ActionArray arrs = {f1, f2, f3};
arrs[0](5);
arrs[1](6);
arrs[2](7);
```

Check: [[C C++ Programming Language - Spiral Rule]]

---

**Aliasing with Templates**
```cpp
// Template Aliase for any pointer
// typedef
template <typename T> class Container {
public:
    typedef T* Ptr;
};
// using
template <typename T> using Ptr = T*;
```

```cpp
// Container<int>::Ptr p = new int(10);
Ptr<int> p = new int(10);
cout << *p << endl;
delete p;
```

---

**`#include <functional>`**

Обертка для функций, lambda, методов.
Это облегчает использование указателей на функции, не нужно думать про Spiral Rule и возиться с raw pointer-ами.

```cpp
#include <functional>

using Callback = std::function<void(int)>;
// using Callback = void(*)(int); // Equivalent

void fun(int a){
    cout << "fun: " << a << endl;
}
```

```cpp
Callback cb = fun;
cb(123);
```

---

**Strange**

```cpp
class MyClass {
public:
    void print(int a){ cout << "MyClass::print: " << a << endl; }
};
```

```cpp
 MyClass obj;
void (MyClass::*ptr)(int) = &MyClass::print;
(obj.*ptr)(42);
```
