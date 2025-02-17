// https://www.w3schools.com/cpp/cpp_exceptions.asp

#include <iostream>

int main(){

    std::cout << "\nExample with try catch: \n" << std::endl;

    try {
            int age = 16;
            if(age < 18){
                // or you can throw error number, i.g. 505
                throw (age);
            }
            std::cout << "Access granted" << std::endl;
    }
    catch (int age){
        std::cout << "Access denied \n";
        std::cout << "Age is: " << age << std::endl;
    }

    std::cout << "\nExample with try catch any: \n" << std::endl;

    // Any
    try {
            int age = 16;
            if(age < 18){
                // or you can throw error number, i.g. 505
                throw (age);
            }
            std::cout << "Access granted" << std::endl;
    }
    catch (...){
        std::cout << "Access denied \n";
    }

}
