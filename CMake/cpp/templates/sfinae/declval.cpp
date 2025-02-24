// https://en.cppreference.com/w/cpp/utility/declval

#include <iostream>
#include <utility>

struct Default
{
    Default(){
        std::cout << "Default's Constructor" << std::endl;
    }
    int foo() const { return 1; }
};

struct NonDefault
{
    NonDefault() = delete;
    int foo() const { return 1; }
};

int main()
{
    decltype(Default().foo())               n1 = 1;     // type of n1 is int
    decltype(std::declval<Default>().foo()) n2 = 1;     // same

//  decltype(NonDefault().foo())               n3 = n1; // error: no default constructor
    decltype(std::declval<NonDefault>().foo()) n3 = n1; // type of n3 is int

    std::cout << "n1 = " << n1 << typeid(n1).name() << '\n'
              << "n2 = " << n2 << typeid(n1).name() << '\n'
              << "n3 = " << n3 << typeid(n1).name() << '\n';
}
