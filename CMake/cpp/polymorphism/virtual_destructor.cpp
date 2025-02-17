#include <iostream>
#include <string>

// --- Named ---
class Named {
protected:
    std::string name;
public:
    Named(std::string name){ this->name = name; }
    ~Named(){
        std::cout << "Destructor call of Named class" << std::endl;
    }
};
// ------

// --- Animal ---
class Animal : public Named {
public:
    Animal(std::string name) : Named(name){}
    virtual ~Animal(){
        std::cout << "Destructor call of Animal class" << std::endl;
    }
};
// ------

// --- Dog ---
class Dog : public Animal {
public:
    Dog(std::string name) : Animal(name) {}
     ~Dog(){
        std::cout << "Destructor call of Dog class" << std::endl;
    }
};
// ------

int main(){
    Animal * dog = new Dog("DogNameP");

    delete dog;
}
