# `auto`, `decltype`, and `type_info`
---

References:
- https://www.geeksforgeeks.org/type-inference-in-c-auto-and-decltype/
- https://en.cppreference.com/w/cpp/language/typeid

- `auto` - in compile-time
- `decltype(expr)` - in compile-time
- `const std::type_info& ti = typeid(var);` - either in compile or in runtime
	- `ti.name()`
	- `ti.hash_code()`

```cpp
#include <iostream>

// --- Data ---
class Data {
private:
    int * __theData;
public:
    Data(int data){
        __theData = new int(data);
    }
    ~Data(){ delete __theData; }
    int get() const { return *__theData; }

    Data() = delete;
    // Data(const Data & data) = delete;
    Data(const Data & data){
        __theData = new int(data.get());
    }
};
// ------

// --- fun ---
auto fun(){ // returns int
    return -1;
}
// ------

// --- min ---
// Возвращает минимальное значение с возвращаемым типом этого минимального значения
template <typename U, typename V> auto min(U u, V v) -> decltype(u < v ? u : v){
    return (u < v) ? u : v;
}
// ------

int main(){
    // --- primitive types ---
    int a = 123; // i
    auto * ap = &a; // Pi

    auto d = 123.; // d
    auto f = 123.f; // f
    auto c = '1'; // c
    auto s = "123"; // PKc ?
    // --- cout ---
    std::cout << typeid(a).name() << std::endl; // i
    std::cout << typeid(ap).name() << std::endl; // Pi

    std::cout << typeid(d).name() << std::endl; // d
    std::cout << typeid(f).name() << std::endl; // f
    std::cout << typeid(c).name() << std::endl; // c
    std::cout << typeid(s).name() << std::endl; // PKc ?
    std::cout << "---" << std::endl;

    // --- array ---
    int arr[]{1, 2, 3}; // A3_i
    std::cout << typeid(arr).name() << std::endl;
    std::cout << "---" << std::endl;

    // --- class ---
    Data data(123); // 4Data (4 letters in the name)
    auto D = data; // Copy of data through Copy constructor
    auto * Dp = &data; // P4Data
    auto ** Dpp = &Dp; // PP4Data
    auto *** Dppp = &Dpp; // PPP4Data

    std::cout << typeid(data).name() << std::endl;
    std::cout << typeid(D).name() << std::endl;
    std::cout << typeid(Dp).name() << std::endl;
    std::cout << typeid(Dpp).name() << std::endl;
    std::cout << typeid(Dppp).name() << std::endl;
    std::cout << "---" << std::endl;

    // ---decltype---
    decltype('A') x; // c
    decltype(fun()) rx; // i, тип возвращаемого значения из функции
    decltype(min(1., 3.f)) mx; // d
    std::cout << typeid(x).name() << std::endl;
    std::cout << typeid(rx).name() << std::endl;
    std::cout << typeid(mx).name() << std::endl;
}
```

PKc - Pointer Constant Character

---

Polymorphic types вычисляются in-runtime:
```cpp
class Base {
public:
    virtual ~Base(){} // Make it Polymorphic adding vtable
};
class Derived : public Base {};
```

```cpp
// --- runtime ---
Derived derived; // 7Derived
Derived* derivedp = &derived; // 7Derived
Base base = derived; // P4Base
Base* basep = &derived; // P4Base ?
Base& baser = derived; // 7Derived - in runtime

std::cout << typeid(derived).name() << std::endl; // 7Derived
std::cout << typeid(derivedp).name() << std::endl; // P7Derived
std::cout << typeid(base).name() << std::endl; // 4Base
std::cout << typeid(basep).name() << std::endl; // P4Base
std::cout << typeid(baser).name() << std::endl; // 7Derived
std::cout << "---" << std::endl;
```

---

Проверить совпадение типов:
```cpp
// --- type_info ---
int A = 234, B = 345;
const std::type_info& ti1 = typeid(A);
const std::type_info& ti2 = typeid(B);

assert(ti1.name() == ti2.name()); // OK
assert(ti1.hash_code() == ti2.hash_code()); // OK

std::cout << ti1.name() << std::endl; // i
std::cout << ti2.name() << std::endl; // i
std::cout << ti1.hash_code() << std::endl; // 6253375586064260614
std::cout << ti2.hash_code() << std::endl; // 6253375586064260614
```
