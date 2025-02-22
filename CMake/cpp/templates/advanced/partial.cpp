#include <iostream>
#include <vector>
#include <type_traits> // std::remove_reference_t

using std::cout, std::endl;

// typename and numeric template
template <typename T, int N>
class Array {
public:
    Array(){
        cout << "Constructor() of Array <" << typeid(T).name() << ", " << N << ">" << endl;
    }
    Array(T){
        cout << "Constructor(T) of Array <" << typeid(T).name() << ", " << N << ">" << endl;
    }
    Array(T*){
        cout << "Constructor(T*) of Array <" << typeid(T).name() << ", " << N << ">" << endl;
    }
    // class template arguments don't need to be repeated it they are the default ones
    Array(const Array & other){
        cout << "Copy Constructor of Array <" << typeid(T).name() << ", " << N << ">" << endl;
    }
};
template <typename T> Array(T) -> Array<T, sizeof(T)>; // deduction guide


// 674 Deduction
template <typename T> class D {
public:
    T t;
    D(T & t){
        cout << "Constructor(T&) of D <" << typeid(T).name() << ">" << endl;
    }
    template <typename R> D(R & r){
        cout << "Constructor(R&) of D <" << typeid(T).name() << ">" << endl;
    }
    // D(T && t){
    //     cout << "Constructor(T&&) of D <" << typeid(T).name() << ">" << endl;
    // }
    template <typename R> D(R && r) : t(std::forward<R>(r)){
        cout << "Constructor(R&&) of D <" << typeid(R).name() << ">" << endl;
    }
};
template <typename R> D(R && r) -> D <std::remove_reference_t<R>&&>;
template <typename R> D(R & r) -> D <double>;
// Alias Template
template <typename R> using Dt = D<R>;


// Deduction Guide - Container
template <typename T> class Container {
public:
    template <typename Iter> Container(Iter beg, Iter end){}
};
template <typename Iter> Container(Iter beg, Iter end) ->
Container<typename std::iterator_traits<Iter>::value_type>;


// Generic class template
template <typename T, typename R>
class A {
public:
    T x;
    A(){
        cout << "Constructor of Generic A <" << typeid(T).name() << ", " << typeid(R).name() << ">" << endl;
    }
};
// Partial specialization
template <typename T>
class A <T, int> {
public:
    T y;
    A(){
        cout << "Constructor of Partial A <" << typeid(T).name() << ", " << "int" << ">" << endl;
    }
};
// Full specialization
template <>
class A <float, int> {
public:
    float z;
    A(){
        cout << "Constructor of Full A <float, int>" << endl;
    }
};


// Type Trait
// #include <type_traits>
template <typename T, typename R>
class is_same {
public:
    static constexpr bool value = false;
};
template <typename T>
class is_same <T, T> {
public:
    static constexpr bool value = true;
};


// Check if pointer is const
// std::true_type and std::false_type contain a field "value" set to true or false respectively
template <typename T> // Generic template declaration
class is_const_pointer : public std::false_type {};
// Partial specialization
template <typename R>
class is_const_pointer<const R*> : public std::true_type {};


// Compare
template <typename T> class B {};

template <typename T, typename R> // Generic template declaration
class Compare : public std::false_type {};

template <typename T, typename R> // Partial specialization
class Compare <B<T>, B<R>> : public std::true_type {};


// Template deduction guide
template <typename T>
class MyString {
public:
    MyString(T){}
    MyString get(){
        return MyString("abc");
    }
};
MyString(char const *) -> MyString<std::string>; // deduction guide
// Factory vs. MyString<const char *> cstr("abc")
template <typename T> auto make_my_string(const T& x){
    return MyString(x);
} // Но никто не гарантирует, что это будет строка


// Driver code
int main(){
    Array<int, 10> _array;
    Array _array_copy = _array; // + type deduction

    int tmp[5];
    cout << sizeof(tmp) << endl;
    Array _deduced_array(tmp);

    // every class specialization is a completely new class
    // and they don't share anything with generic class
    A<float, float> ga;
    A<int, int> pa;
    A<float, int> fa;

    // ---
    cout << "---" << endl;
    cout << std::boolalpha;
    cout << is_same<int, char>::value << endl;
    cout << is_same<float, float>::value << endl;

    // ---
    cout << "---" << endl;
    cout << is_const_pointer<int>::value << endl;
    cout << is_const_pointer<int*>::value << endl;
    cout << is_const_pointer<const int*>::value << endl;

    // ---
    cout << "---" << endl;
    cout << Compare<int, float>::value << endl;
    cout << Compare<B<int>, B<float>>::value << endl;

    // ---
    cout << "---" << endl;
    const char * cstr = "abc";
    // MyString sptr(cstr);
    // MyString sptr={cstr};
    MyString sptr = make_my_string(cstr);
    MyString<const char *> csptr={cstr};
    std::cout << typeid(sptr).name() << std::endl;
    std::cout << typeid(sptr.get()).name() << std::endl;
    std::cout << typeid(csptr).name() << std::endl;
    std::cout << typeid(csptr.get()).name() << std::endl;

    // ---
    cout << "---" << endl;
    int a = 10;
    // D d(a);
    D d(std::move(a));
    std::cout << typeid(d).name() << std::endl;
    std::cout << typeid(d.t).name() << std::endl;
    // int&& aref = 20;
    // std::cout << typeid(aref).name() << std::endl;
    D di(a);
    Dt dti(a);

    // ---
    cout << "---" << endl;
    std::vector v {1, 2, 3};
    Container container {v.begin(), v.end()};

}
