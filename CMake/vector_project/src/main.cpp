#include <iostream>
#include "mymath/mymath.h"

using std::cout, std::endl;

using namespace mymath;
using namespace mymath::myvector;

int main(int argc, char * argv[]){
    assert(2 + 2 == 4 && "Assertion violated!");

    NamedObject namedObject;

    cout << "\n";
    // malloc and free
    // NamedObject * ptr_vector = new Vector(5);
    Vector * ptr_vector = new Vector(5);
    ptr_vector->setName("My Vector");
    for(int i = 0; i < ptr_vector->getSize(); i++){
        // (*ptr_vector)[i] = i;
        ptr_vector->set(i+1, i);
    }

    // NamedObject vector = *ptr_vector;
    // vector.sayHello(); // Здесь вызовется функция супер-класса
    ptr_vector->sayHello();
    ptr_vector->printCoeffs();


    cout << "\n";
    Vector copied_vector(*ptr_vector);
    copied_vector.setName("Copied Vector");
    copied_vector.sayHello();
    copied_vector.printCoeffs();


    cout << "\n *= \n";
    copied_vector *= *ptr_vector;
    ptr_vector->printCoeffs();
    copied_vector.printCoeffs();


    cout << "\n * \n";
    Vector mul = copied_vector * (*ptr_vector);
    ptr_vector->printCoeffs();
    copied_vector.printCoeffs();
    mul.printCoeffs();


    cout << "\n---Destructors---" << endl;

    delete ptr_vector;

    return EXIT_SUCCESS;
}


// int FOOBAR;
// int FOOBAR; // ERROR, previously declared
extern int FOOBAR;
extern int FOOBAR;
// FOOBAR = 123; // нельзя присваивать вне функции или вне определения

void global(); // extern by default
void global();
void global(){
    cout << "Global!" << "\n";
}

/*
 * Local scope
 * В C++ рекомендуется использовать anonymous namespace
 *
 * В C сделать local можно было с static
 * static memory allocation is allocated at startup and never changes
 * Область видимости static memory ограничена файлом

 static void local(){ cout << "Static Lol!" << "\n"; }

 */
namespace {
    void local(){
        cout << "Local!" << "\n";
    }
}
