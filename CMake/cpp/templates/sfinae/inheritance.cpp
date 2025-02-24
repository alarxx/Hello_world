#include <iostream>

template <typename T> class A {
public:
    T x;
    A(T t) : x(t) {}
    void f(){ std::cout << "Classic A" << std::endl; }
};

// Specialization
template <> class A<float> {
public:
    float x;
    A(float t) : x(t) {}
    void f(){ std::cout << "Float A" << std::endl; }
};

template <typename T>
class B : A<T> {
public:
    using A<T>::x;
    using A<T>::f;

    B(T t) : A<T>(t) {}

    void g(){
        std::cout << x << std::endl;
        f();
    }
};

int main(){
    B b(1.f);
    b.f();
}
