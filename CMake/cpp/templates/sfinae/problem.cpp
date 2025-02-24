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
    print_size(x);  // Ошибка! У int нет size()
}

