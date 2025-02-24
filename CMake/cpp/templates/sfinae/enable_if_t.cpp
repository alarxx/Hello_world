#include <iostream>
#include <type_traits>

using std::cout, std::endl;

template <typename T>
std::enable_if_t<std::is_signed_v<T>, void>
f(T t){
    cout << "signed" << endl;
}

template <typename T>
std::enable_if_t<!std::is_signed_v<T>>
f(T t){
    cout << "unsigned" << endl;
}

int main(){
    f(1);
    f(1u);
}
