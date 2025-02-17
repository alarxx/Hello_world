/*
 * - https://www.w3schools.com/cpp/cpp_polymorphism.asp
 * - https://www.geeksforgeeks.org/cpp-polymorphism/
 * - https://cplusplus.com/doc/tutorial/polymorphism/
 */

#pragma once
#ifndef _MY_CLASSES_H_
#define _MY_CLASSES_H_

#include <iostream>
#include <string>
#include <memory>

// --- Named ---

// Abstract class
// Классы продолжают быть абстрактными, пока не реализуют все pure virtual functions
class Named {
protected:
    std::string name;
public:
    Named(std::string name){
        // variable shadowing
        this->name = name;
    }
    // Pure virtual function
    virtual void introduce() = 0;
    // virtual void introduce(); // undefined reference to `vtable for Named'
};

// ------

// --- Animal ---

class Animal : public Named {
public:
    Animal(std::string name) : Named(name){}
    virtual void makeSound(){
        std::cout << name << ": Animal sound!" << std::endl;
    }
    virtual void introduce(){
        std::cout << "Hello, my name is " << name << std::endl;
    }
};

// ------

// --- Dog ---

// Multiple inheritance exists
// https://www.geeksforgeeks.org/cpp-inheritance-access/
// По умолчанию private наследование, то есть все поля становятся private
// Если выбрать protected, то все public и protected поля станут protected
// public ничего не меняет
class Dog : public Animal {
public:
    Dog(std::string name) : Animal(name) {}
    //  ~Dog(){
    //     std::cout << "Destructor call of Dog class" << std::endl;
    // }
    // Definition of the member function of the base class
    void makeSound() override {
        std::cout << name << ": Bark bark!" << std::endl;
    }
    void bark(){
        std::cout << name << ": Bark bark!" << std::endl;
    }
};

// ------

// --- Cat ---

class Cat : public Animal {
public:
    Cat(std::string name) : Animal(name) {}
    // Definition of the member function of the base class
    void makeSound() override {
        std::cout << name << ": Meow meow!" << std::endl;
    }
    void meow(){
        std::cout << name << ": Meow meow!" << std::endl;
    }
};

// ------

#endif
