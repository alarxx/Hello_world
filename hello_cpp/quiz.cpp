#include <iostream>

void increment(int x) {
    x++;
}

int main() {
    int num = 7;
    increment(num);
    std::cout << num;
    return 0;
}
