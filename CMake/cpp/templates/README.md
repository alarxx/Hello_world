
# `Template <typename T>`

https://www.geeksforgeeks.org/templates-cpp/

----

#### Function Template
```c++
// typename keyword is equivalent to a class, but not always
template <typename T> T square(T v){
    return v * v;
}
```

#### Template Specialization

https://www.geeksforgeeks.org/template-specialization-c/

```cpp
// Template Specialization: for char data type
template <> char square<char>(char v){
// template <> char square(char v){ // Необязательно уточнять <char>
    return 'N';
}
```

---

#### `typename` vs. `class`

typename keyword is equivalent to a class, but not always.

template template

Шаблонный класс принимающий в виде типа шаблонный класс:
```cpp
#include <iostream>

// Получается это не класс, а шаблон
template <typename T> class A {
public:
    T data;
    // Initializer list
    A(T data) : data(data){}
};

// typename vs. class
// template <template <typename> typename T, typename S = char> class B { // invalid!
template <template <typename> class T, typename S = char> class B { // valid!
// template <class T, typename S> class B { // Compilation Error! Because we use T as template by itself
public:
    T<S> obj; // A<int>
    B(S value) : obj(value) {} // B(int value){ obj(value); }
    S get(){
        return obj.data;
    }
};

int main(){
    B<A, int> obj(10);

    auto value = obj.get();
    std::cout << value << std::endl;
}
```

---

#### Default Arguments and Non-Type Parameters

```cpp
template <typename T = int, int tsize = 0> class Array {
	...
};
```

Если не указать T, то он по умолчанию int, если не указать tsize, то он по умолчанию 0.
```cpp
// Default template values <int, 0>
Array defaultArr;
// Empty string
defaultArr.print();
```

---

#### Argument Deduction
При передаче шаблонного типа через аргумент конструктора можно не указывать тип
```cpp
template <typename T> class Data {
private:
    T __theData;
public:
    Data(){}
    Data(T data) : __theData(data) {}
    void set(T data){ __theData = data; }
    T get(){ return __theData; }
};

```

Basic use:
```cpp
Data<int> dataObj1;
dataObj1.set(101);
std::cout << "data object 1: " << dataObj1.get() << std::endl;
```

Argument Deduction - можно опустить указание типа:
```cpp
Data dataObj2(202);
std::cout << "data object 2: " << dataObj2.get() << std::endl;
```

---

## Nice Syntax with initializer_list

https://www.geeksforgeeks.org/std-initializer_list-in-cpp-11/

Что мы хотим реализовать:
```cpp
// Array<int, 5> arr = {1, 2, 3, 4};
Array arr = {1, 2, 3, 4};
```

`initializer_list` подходит только для Nice Syntax, и нужно копировать массив, потому что он локально (in Stack) выделяет массив in contiguous памяти, то есть значения нельзя менять, и после выходы из функции массив обязательно удаляется.

Реализация через `initializer_list`:
```cpp
#include <initializer_list>

template <typename T = int, int tsize = 0> class Array {
private:
    int __size;
    T * __coeffs;
public:
    Array(std::initializer_list<T> list);
    ~Array(){ delete[] __coeffs; }
    int size() const { return __size; }
    T * data(){ return __coeffs; }
    void set(int i, T x) { __coeffs[i] = x; }
    int get(int i) const { return __coeffs[i]; }
    void print();
};

...

template <typename T, int tsize> Array<T, tsize>::Array(std::initializer_list<T> list){
	//  На случай если мы не передадим tsize (Argument Deduction)
    __size = tsize ? tsize : list.size();
    __coeffs = new T [list.size()];

    // ---
    // const int * array = list.begin(); // мы могли бы так сделать, но ...
    // 1) сами значения array [i] были бы const
    // 2) initializer_list in Static Memory, so it will be deleted after function completion what will lead to dangling pointer
    // ---
    // So we create copy
    int i = 0;
    // auto
    // for(auto it = list.begin(); it != list.end(); it++, i++){
    // std::initializer_list<T>::iterator // Тут почему-то нельзя использовать из-за шаблонного initializer_list
    // typename initializer_list<int>::iterator
    for(typename std::initializer_list<T>::iterator it = list.begin(); it != list.end() /*&& i < list.size()*/; it++, i++){
        __coeffs[i] = *it;
    }
}
```


## Advanced

### Partial specialization

<- [[federico - Modern CPP Programming#Partial specialization]]

```cpp
// Generic class template
template <typename T, typename R>
class A {
	T x;
};

// Partial specialization
template <typename T>
class A <T, int> {
	T y;
};

// Full specialization
template <>
class A <float, int> {
	T z;
};
```

==Нельзя создавать partially specialized class и specialized function.==
Класс должен быть specialized, либо оба generic, либо оба fully specialized.
```cpp
template <typename T, typename R>
class A {
public:
	template <typename X, typename Y> void fun();
};


template <>
template <typename X, template Y>
void A<int, int>::fun<X, Y>(){ // OK
	...
}

template <typename T, typename R>
template <>
void A<T, R>::fun<int, int>(){ // Error
	...
}
```

---

### Dependent name

`a.template g<int>()`

Здесь `template` говорит, что то что последует является шаблоном (function or class):
```cpp
#include <iostream>

template <typename T> class A {
public:
    template <typename R> void g(){
        std::cout << "T: " << typeid(T).name() << std::endl;
        std::cout << "R: " << typeid(R).name() << std::endl;
    }
};

template <typename T> void f(A<T> a){
    // a.g<int>(); // (a.g < int) > ()
    // Здесь template говорит, что то что последует является шаблоном (function or class)
    a.template g<int>();
}

int main(){
    A<float> a;
    f(a);
}
```

---

##### Inheritance

```cpp
template <typename T> class A {
public:
    T x;
    A(T t) : x(t) {}
    void f(){ cout << "Classic A" << endl; }
};

template <> class A<float> {
public:
    float x;
    A(float t) : x(t) {}
    void f(){ cout << "Float A" << endl; }
};

template <typename T>
class B : A<T> {
public:
	using A<T>::x; // Иначе компилятор не понимает что за x
	using A<T>::f;

    B(T t) : A<T>(t) {}

    void g(){
        cout << x << endl;
        f();
    }
};
```

```cpp
B b(1.f);
b.g();
// 1
// Float A
```

---
### `using type = T`

Template класс хранит его тип в поле, к которому можно получить доступ.

Мы можем хранить сам тип класса, и потом создавать переменные с type of этого класса:
```cpp
#include <iostream>

template <typename T>
class A {
public:
    using type = T; // the type of class
};

template <typename T>
using AType = typename A<T>::type;

template <typename T>
void fun(A<T> a){
    AType<T> var;
    // typename A<T>::type var;
    // int var;
    std::cout << typeid(var).name() << std::endl; // i
}
```

```cpp
A<int> ai;
std::cout << typeid(ai).name() << std::endl; // 1AIiE
fun(ai);
```

---
### Template deduction guide

`MyString(char const *) -> MyString<std::string>;`

Если мы явно не укажем тип при создании объекта класса, то будет происходить автоматический deduction, но мы можем указать **deduction guide**, в котором указываем какой тип должен быть, если придет определенный тип в конструктор:
```cpp
template <typename T>
class MyString {
public:
    MyString(T t){
	    std::cout << "Constructor: " << typeid(t).name() << std::endl;
	    // PKc
	    // NSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
    }
    MyString get(){
        return MyString("abc");
        // return MyString<T>("abc");
    }
};

// Deduction guide
MyString(char const *) -> MyString<std::string>;

// Factory vs. MyString<const char *> cstr("abc")
template <typename T> auto make_my_string(const T& x){
    return MyString(x);
    // deduction guide make:
    // return MyString<std::string>(x);
}
```

Когда мы создадим `MyString` без указания типа и передадим `const char *`, то будет `std::string` due to **deduction guide**:
```cpp
const char * cstr = "abc";

MyString csptr(cstr); // string
// MyString csptr={cstr}; // string
// MyString csptr = cstr; // error, присваивание с deduction к шаблону не поддерживается
// MyString<const char *> csptr = cstr; // const char *
```

Но если мы explicitly укажем тип, то deduction guide естественно не сработает:
```cpp
MyString<const char *> csptr(cstr); // const char *
```

И если мы хотим, чтобы всегда было `<std::string>` мы можем использовать factory:
```cpp
MyString sptr = make_my_string(cstr);
```
Теперь мы не можем explicitly указывать тип:
```cpp
// MyString<const char *> sptr = make_my_string(cstr);
// Error, <string> to <const char *>
```

---
### Compile-time recursive

[[Meta Programming]]

Template meta-programming is fast in runtime, nothing is computed at runtime,
it is computed at compile-time so it may slow down compilation time.
###### Compile-time Factorial Example

Class
```cpp
template <int N>
class Factorial {
public:
    static constexpr int number = N * Factorial<N - 1>::number;
};

// Base case
template <>
class Factorial<1>{
public:
    static constexpr int number = 1;
};
```

Function
```cpp
template <typename T> constexpr T factorial(const T N){
    T tmp = N;
    for(int i=2; i<N; i++){
        tmp *= i;
    }
    return tmp;
}
```

```cpp
// constexpr int number = Factorial<5>::number;
constexpr int number = factorial<int>(5);
cout << number << endl; // 120
```

##### Compile-time Log Example

```cpp
template <int A, int B>
class Max {
public:
    static constexpr int value = A > B ? A : B;
};

template <int N, int BASE>
class Log {
public:
    static_assert(N > 0, "Number must be greater than zero");
    static_assert(BASE > 0, "Base must be greater than zero");

    static constexpr int TMP = Max<1, N / BASE>::value;
    static constexpr int value = 1 + Log<TMP, BASE>::value;
};

// Base case
template <int BASE>
class Log<1, BASE> {
public:
    static constexpr int value = 0;
};
```

```cpp
constexpr int log2_20 = Log<20, 2>::value;
cout << log2_20 << endl; // 4
// 10 = 1
// 5 = +1
// 2.5 = +1
// 1.25 = +1
// 1 = +0
```

##### Unroll example

Compile-time / Runtime mix
Не совсем понятно, как будто бы это все происходит in runtime.
```cpp
template <int NUM_UNROLL, int STEP = 0>
class Unroll {
public:
    template <typename Op>
    static void run(Op op){
        op(STEP);
        Unroll<NUM_UNROLL, STEP + 1>::run(op);
    }
};

// Base case
template <int NUM_UNROLL>
class Unroll<NUM_UNROLL, NUM_UNROLL> {
public:
    template <typename Op>
    static void run(Op op){
        op(NUM_UNROLL);
    }
};
```

```cpp
auto lambda = [](int n){
	cout << n << endl;
};
Unroll<5>::run(lambda);
```

---


### SFINAE

[[Substitution Failure Is Not An Error (SFINAE)]]
#### Brief about SFINAE

**SFINAE** - это Compile-time проверка подходит ли тип под специализацию шаблона.
Специализированные шаблоны которые не подходят под переданный тип исключаются и переходят на следующий или base, а не выкидывают ошибку.

Мы использум SFINAE, если хотим создать специализацию только под определенный тип и не разрешать использовать неподходящие.

SFINAE работает только в случае подстановки шаблонного параметра.
То есть это нужно чтобы компилятор мог выбрать какой из шаблонов выбрать.

- `enable_if<Condition, Type>`
- `Condition<T, expression(T) = void> : true_type | false_type` `::value`

```cpp
template <bool Condition, template Type>
class enable_if {};

// Specialization
template <typename Type>
class enable_if<true, Type> {
public:
	static constexpr type = Type;
};

template <bool Condition, template Type>
using enable_if_t = enable_if<Condition, Type>::type;
```

Для SFINAE очень важно понимание Template Specialization.
Нам не хватит одного using:
```cpp
template <typename T>
using has_size = decltype(declval<T>().size(), true_type/*false?*/);
```

`Condition`:
```cpp
// Default Base
template <typename T, typename U = void>
class has_size : false_type {};
// U нужен только для проверки T, он по итогу всегда void

// Specialization
template <typename T>
class has_size<T, decltype( ? , void())> : true_type {};

template <typename T>
using has_size_v = has_size<T>::value;
```

На месте ? мы должны вписать проверку:
```cpp
decltype(declval<T>().size(), void())
```

Using:
```cpp
template <typename T>
enable_if_t<has_size_v<T>, T>
fun(T t){...}

fun(string("Hello")); // Ok
fun(vector({1, 2, 3})); //Ok
fun(42.f); // Compile-time Error
```

---

#### SFINAE Problem

```cpp
#include <iostream>
#include <string>

template <typename T>
void print_size(const T& obj) {
    std::cout << obj.size() << std::endl;
}

int main() {
    std::string s = "Hello";
    print_size(s);  // Работает, у std::string есть size()

    int x = 42;
    print_size(x);  // Compilation Error! У int нет size()
}
```
У `int` нет `size()`, у нас выходит ошибка компиляции.

Как можно проверить и указать правильную специализацию шаблона?

---
#### Template Specialization
irrelevant in SFINAE context.
В примере ниже мы declare-им template функцию,
потом определяем specialization для int и unsigned int,
И дедукция работате для этих функций, но ==проблема возникнет, когда мы заходим передать не специализированные типы int и unsigned int==: long, long long, float, double, etc :
```cpp
#include <iostream>
using std::cout, std::endl;

template <typename T>
T ceil_div(T value, T div);

template <>
unsigned ceil_div<unsigned>(unsigned value, unsigned div){
    return (value + div - 1) / div;
}

template <>
int ceil_div<int>(int value, int div){
    // XOR bitwise operator
    // Если один из них меньше 0, то результат будет с минусом:
    // (-5/2)=-2.5; ceil(-2.5)=-2
    return (value > 0) ^ (div > 0) ?
	    (value / div) :
	    (value + div - 1) / div;
}

int main(){
    int c = ceil_div(8, 2); // Ok
    cout << c << ": " << typeid(c).name() << endl;

    unsigned u = 10;
    cout << u << ": " << typeid(u).name() << endl;

    unsigned uu = ceil_div(8u, 2u); // Ok
    // unsigned long luu = ceil_div(8lu, 2lu); // Compilation Error: undefined reference to `unsigned long ceil_div<unsigned long>(unsigned long, unsigned long)'
    unsigned long luu = ceil_div<int>(8lu, 2lu); // Ok, without deduction, explicitly setting to int
}
```

---

#### What is SFINAE?
**SFINAE** - это Compile-time проверка подходит ли тип под специализацию шаблона.
Шаблоны которые не подходят под переданный тип исключаются и переходят на следующий или base, а не выкидывают ошибку.

---
#### `enable_if`:
Если мы передаем в enable_if правильный expression, то его type будет тот, который мы передали, но если expression false, то type is undefined и у нас выходит ошибка:
```cpp
#include <iostream>

template <bool Condition, typename T = void>
class enable_if {
// type is not defined of Condition == false
};

// Partial Specialization
template <typename T>
class enable_if<true, T> {
public:
	using type = T;
};
```

`enable_if_t` - is an alias of `typename enable_if<>::type`
```cpp
template <bool Condition, typename T>
using enable_if_t = typename enable_if<Condition, T>::type;
```

```cpp
enable_if_t<2+2==4, int> a = (float) 42.f; // int a = 42

cout << a << ": " << typeid(a).name() << endl; // 42: i
```

---

```cpp
template <typename T>
enable_if_t<std::is_floating_point_v<T>, float>
fun(T t){
    cout << "float fun" << endl;
    return t;
}

template <typename T>
enable_if_t<std::is_integral_v<T>, int>
fun(T t){
    cout << "int fun" << endl;
    return t;
}
```

```cpp
fun(1); // int fun
fun(1.f); // float fun
fun(std::string("Hello")); // Error
```

---

#### Как написать свою проверку?

```cpp
#include <type_traits>
using std::void_t;
using std::declval;

template <typename T, typename U = void>
class has_size {
public:
	static constexpr bool value = false;
	// Либо можно наследоваться от std::false_type
};

// Template Partial Specialization
template <typename T>
class has_size<T, void_t<decltype(declval<T>().size()>> {
public:
	static constexpr bool value = true;
	// Либо можно наследоваться от std::true_type
};

// Alias of has_size<T>::value
template <typename T>
inline constexpr bool has_size_v = has_size<T>::value;
```

`std::void_t< decltype(std::declval<T>().size())` ??

[[С++ Programming Language - auto]]

**[`decltype(entity)`](https://en.cppreference.com/w/cpp/utility/declval)** - returns type in-compile time
```cpp
decltype('A') x; // char x
```

**[`declval<T>()`](https://en.cppreference.com/w/cpp/utility/declval)** - makes instance of T and makes possible to use member functions of T. Используется только in unevaluated context, типично с decltype.
```cpp
#include <utility>
decltype(Default().foo())               n1 = 1; // type of n1 is int
decltype(std::declval<Default>().foo()) n2 = 1; // same
```
При этом он не вызывает конструктор класса Default, это все происходит как бы во время компиляции. Так же и не вызывается метод foo().

[[C++ Programming Language - comma operator]]

**[`void_t`](https://en.cppreference.com/w/cpp/types/void_t)**
```cpp
template <typename ... >
using void_t = void;
```

То есть нам без разницы на U в has_size.

---

```cpp
template <typename T>
enable_if_t<has_size_v<T>, std::string>
fun(T t){
	cout << "string fun" << endl;
    return t;
}
```

---

#### integral_constant: true_type, false_type
https://en.cppreference.com/w/cpp/types/integral_constant
```cpp
template <typename T, T v>
class integral_constant {
public:
	static constexpr T value = v;
	using value_type = T;
	// using type = integral_constant<T, v>;
	// constexpr operator T(){ return v; }
};
```

`true_type` and `false_type` classes:
```cpp
using false_type = integral_constant<bool, false>;
using true_type = integral_constant<bool, true>;
```

---

#### SFINAE is about Templates Specialization

SFINAE работает только в случае подстановки шаблонного параметра.
То есть это нужно чтобы компилятор мог выбрать какой из шаблонов выбрать.

```cpp
template <typename T>
struct has_size {
private:
    // Trailing return type: auto fun() -> int;
    static constexpr auto test(int) ->
	    decltype(declval<T>().size(), true_type());
	static constexpr false_type test(...);
public:
    static constexpr bool value = decltype(test(0))::value;
};

int main() {
    has_size<string>::value;  // true
    has_size<int>::value;  // Runtime Error
}
```
Здесь получается `.size()` не существует и во время компиляции выходит ошибка подстановки (substitution).

Вот так SFINAE будет работать:
```cpp
template <typename T>
struct has_size {
private:
    // Trailing return type: auto fun() -> int;
    template <typename U> // without it it won't work
    static constexpr auto test(int) ->
	    decltype(declval<U>().size(), true_type());
    template <typename>
    static constexpr false_type test(...);
public:
    static constexpr bool value = decltype(test<T>(0))::value;
};

int main() {
    has_size<string>::value;  // true
    has_size<int>::value;  // false
}
```

---
