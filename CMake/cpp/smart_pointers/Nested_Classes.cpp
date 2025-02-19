
// https://www.geeksforgeeks.org/nested-classes-in-c/

#include <iostream>

class Outer {
private:
    static int __x;
public:
    static void setX(int x){
        __x = x;
    }
    class Nested {  // Этот класс не зависит от Outer
    public:
        static void print() {
            std::cout << "nested: " << __x << std::endl;
        }
    };
    static void print() {
        std::cout << "outer: " << __x << std::endl;
    }
};

int Outer::__x = 0;


int main(){
    Outer::setX(321);
    Outer::Nested::print();
    Outer::print();

    return 0;
}
