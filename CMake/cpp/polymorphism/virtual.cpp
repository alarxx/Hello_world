#include "myclasses.h"

int main(){
    // --- Abstract class ---

    // Нельзя создать instance абстрактного класса
    // Named named;
    // error: cannot declare variable ‘named’ to be of abstract type ‘Named’
    // note:   because the following virtual functions are pure within ‘Named’

    // ------

    // --- Animal ----

    Animal animal("AnimalName");
    animal.introduce();
    animal.makeSound();

    // ------

    // --- Dog обрезанный до Animal ----

    Animal dog = Dog("DogName");
    // Animal имеет доступ к `introduce()`
    dog.introduce();
    // Early binding - function is selected at compile time.
    dog.makeSound(); // "Animal sound!", even with virtual function
    // dog.bark();
    // error: ‘class Animal’ has no member named ‘bark’

    // ------

    // --- Pointer of type Animal pointing to Dog object----

    Animal * dogptr = new Dog("DogNameP");
    // std::unique_ptr<Animal> dogptr = std::make_unique<Dog>("DogNameP"); // Smart Pointer
    Animal & dogref = *dogptr;
    // dogptr->introduce();
    dogref.introduce();
    // Late binding (dynamic dispatch) - function is selected in runtime.
    // dogptr->makeSound(); // "Bark bark!"
    dogref.makeSound(); // "Bark bark!"
    // dogptr->bark();
    // error: ‘class Animal’ has no member named ‘bark’

    // ------

    delete dogptr;

    return 0;
}
