/*
 Comma operator

 Syntax:
    `a, b;`
    - a будет вычисляться первым
    - b вычисляется вторым и его значение возвращается

 */

#include <iostream>

void fun(){
    std::cout << "fun" << std::endl;
}

int main(){
    int a = (fun(), 2);
    std::cout << a << std::endl;
}
