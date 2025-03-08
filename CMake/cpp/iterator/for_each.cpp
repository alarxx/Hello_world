#include <iostream>
#include <algorithm>

void print(int a){
    std::cout << a << " ";
}

class MyClass { // Вызываем class как функцию
public:
    void operator () (int a){
        std::cout << a << " ";
    }
};

void exex(int a){
    if(a % 2 == 0){
        throw a;
    }
}

int main(){
    int arr[]{1, 2, 3, 4, 5, 6};
    int size = sizeof(arr) / sizeof(arr[0]);

    // ---

    std::for_each(arr, arr + size, print);

    std::cout << std::endl;

    // ---

    MyClass obj;
    std::for_each(arr, arr + size, obj);

    std::cout << std::endl;

    // ---

    try{
        std::for_each(arr, arr + size, exex);
    }
    catch(int a){
        std::cout << "Handled: " << a << std::endl;
    }

    // ---

    // lambda expression
    std::for_each(arr, arr + size, [](int a)->void{
        std::cout << "lambda: " << a << std::endl;
    });

}
