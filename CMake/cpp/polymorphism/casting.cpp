#include "myclasses.h"

int main(){

/* C++
     https://stackoverflow.com/questions/28002/regular-cast-vs-static-cast-vs-dynamic-cast
     https://stackoverflow.com/questions/332030/when-should-static-cast-dynamic-cast-const-cast-and-reinterpret-cast-be-used

     - C-style type casting (static + dynamic + reinterpret_cast + const_cast)
     - static_cast, runtime cast errors is undefined behaviour
     - dynamic_cast, additionally checking in runtime and can return nullptr or throw std::bad_cast
     - reinterpret_cast, straightforward
     - const_cast, взлом const, кстати еще можно pointer-ами взломать
*/

    Animal * dogptr = new Dog("DogNameP");
    /*
    // C-style casting works
    void * casted_dogptr = (void *) dogptr;
    ((Animal *) casted_dogptr)->makeSound();
    ((Dog *) casted_dogptr)->bark();
    // This works too!
    (static_cast<Cat *>(dogptr))->meow(); // Meow meow!
    */

    // Мы здесь точно знаем, что Dog это Animal и т.д., если бы мы не были уверены, i.g. (Dog *) catptr, то лучше использовать dynamic_cast
    // --- Pointer ---
    // `static_cast` нам бы разрешил:
    // Cat * casted_dogptr = dynamic_cast<Cat *>(dogptr);
    Dog * casted_dogptr = dynamic_cast<Dog *>(dogptr);
    if(!casted_dogptr){
        std::cout << "cast failed!" << std::endl;
        return 1;
    }
    casted_dogptr->makeSound();

    delete dogptr;
    dogptr = nullptr;
    // ------


    // --- Reference ---
    // Dog * catptr = (Dog *) new Cat("CatNameP"); // powerful
    Animal * catptr = new Cat("CatNameP");
    try{
        Dog & catref = dynamic_cast<Dog &>(*catptr); // throws std::bad_cast
        catref.makeSound();
    }
    // https://en.cppreference.com/w/cpp/types/bad_cast
    catch (const std::bad_cast& e){
        std::cout << "Error handled: e.what(): " << e.what() << '\n';
    }
    std::cout << "After try!" << std::endl;
    delete dogptr;
    dogptr = nullptr;
    // ------

    return 0;
}
