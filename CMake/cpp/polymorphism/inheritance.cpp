// https://www.w3schools.com/cpp/cpp_polymorphism.asp

#include <iostream>
#include <string>

class Animal {
public:
    std::string name;
    Animal(std::string name){
        // variable shadowing
        this->name = name;
    }
    void makeSound(){
        std::cout << name << ": Animal sound!" << std::endl;
    }
};

// Multiple inheritance exists
// https://www.geeksforgeeks.org/cpp-inheritance-access/
// По умолчанию private наследование, то есть все поля становятся private
// Если выбрать protected, то все public и protected поля станут protected
// public ничего не меняет
class Dog : public Animal {
public:
    Dog(std::string name) : Animal(name) {}
    // Definition of the member function of the base class
    void makeSound(){
        std::cout << name << ": Bark bark!" << std::endl;
    }
};

int main(){
    Animal animal("AnimalName");
    animal.makeSound();

    Animal dog = Dog("DogName");
    dog.makeSound();

    Animal * dogp = new Dog("DogName");
    dogp->makeSound();
}
