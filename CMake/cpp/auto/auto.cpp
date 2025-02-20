#include <iostream>
#include <cassert>

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

class Base {
public:
    virtual ~Base(){}
};
class Derived : public Base {};

auto fun(){ // returns int
    return -1;
}

// Возвращает минимальное значение с возвращаемым типом этого минимального значения
template <typename U, typename V> auto min(U u, V v) -> decltype(u < v ? u : v){
    return (u < v) ? u : v;
}

int main(){
    // --- primitive types ---
    int a = 123;
    auto * ap = &a;

    auto d = 123.;
    auto f = 123.f;
    auto c = '1';
    auto s = "123";
    // --- cout ---
    std::cout << typeid(a).name() << std::endl;
    std::cout << typeid(ap).name() << std::endl;

    std::cout << typeid(d).name() << std::endl;
    std::cout << typeid(f).name() << std::endl;
    std::cout << typeid(c).name() << std::endl;
    std::cout << typeid(s).name() << std::endl;
    std::cout << "---" << std::endl;


    // --- array ---
    int arr[]{1, 2, 3};
    std::cout << typeid(arr).name() << std::endl;
    std::cout << "---" << std::endl;

    // --- class ---
    Data data(123);
    auto D = data; // Copy of data through Copy constructor
    auto * Dp = &data;
    auto ** Dpp = &Dp;
    auto *** Dppp = &Dpp;

    std::cout << typeid(data).name() << std::endl;
    std::cout << typeid(D).name() << std::endl;
    std::cout << typeid(Dp).name() << std::endl;
    std::cout << typeid(Dpp).name() << std::endl;
    std::cout << typeid(Dppp).name() << std::endl;
    std::cout << "---" << std::endl;

    // --- runtime ---
    Derived derived;
    Derived* derivedp = &derived;
    Base base = derived;
    Base* basep = &derived;
    Base& baser = derived;
    std::cout << typeid(derived).name() << std::endl;
    std::cout << typeid(derivedp).name() << std::endl;
    std::cout << typeid(base).name() << std::endl;
    std::cout << typeid(basep).name() << std::endl;
    std::cout << typeid(baser).name() << std::endl;
    std::cout << "---" << std::endl;

    // ---decltype---
    decltype('A') x;
    decltype(fun()) rx; // тип возвращаемого значения из функции
    decltype(min(1., 3.f)) mx;
    std::cout << typeid(x).name() << std::endl;
    std::cout << typeid(rx).name() << std::endl;
    std::cout << typeid(mx).name() << std::endl;
    std::cout << "---" << std::endl;

    // --- type_info ---
    int A = 234, B = 345;
    const std::type_info& ti1 = typeid(A);
    const std::type_info& ti2 = typeid(B);
    assert(ti1.name() == ti2.name());
    assert(ti1.hash_code() == ti2.hash_code());
    std::cout << ti1.name() << std::endl;
    std::cout << ti2.name() << std::endl;
    std::cout << ti1.hash_code() << std::endl;
    std::cout << ti2.hash_code() << std::endl;

}
