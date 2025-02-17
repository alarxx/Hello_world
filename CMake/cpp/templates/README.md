
# Template <typename T>

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
