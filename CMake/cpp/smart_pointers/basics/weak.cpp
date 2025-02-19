
// https://stackoverflow.com/questions/12030650/when-is-stdweak-ptr-useful
// https://stackoverflow.com/questions/5671241/how-does-weak-ptr-work
// https://en.cppreference.com/w/cpp/memory/shared_ptr - Control Block

#include <iostream>
#include <memory>

#include "Foo.hpp"

/*
 Два shared_ptr ссылающиеся друг на друга никогда не удалятся -> Memory Leak.
*/

int main(){
    /*
    // OLD, problem with dangling pointer
    // PROBLEM: ref will point to undefined data!
    int* ptr = new int(10);
    int* ref = ptr;
    delete ptr;
    */

    std::shared_ptr<Foo> ptr1 = std::make_shared<Foo>();
    std::cout << "use count: " << ptr1.use_count() << std::endl;
    ptr1->show();
    std::cout << "---initial created.---" << std::endl;
    std::cout << std::endl;

    // ------

    // Как отследить, что ptr1 удалился?
    // Мы не можем просто создать другой shared_ptr, чтобы отслеживать ptr1
    // std::shared_ptr<Foo> ptr2 = ptr1; // No
    std::weak_ptr<Foo> weak1 = ptr1;
    std::shared_ptr<Foo> ptr2 = ptr1;

    std::cout << "use count: " << ptr1.use_count() << std::endl;

    ptr1.reset(new Foo);
    // Но так же обычно не делают, меняют значение на которое указывает указатель, а не полностью объект
    std::cout << "use count: " << ptr1.use_count() << std::endl;

    std::weak_ptr<Foo> weak2 = ptr1;

    std::cout << std::endl;

    // ------

    // expired() лучше не использовать и предпочесть lock()
    // weak1 is expired!
    if(auto tmp = weak1.lock())
        tmp->show();
    else
        std::cout << "weak1 is expired\n";

    std::cout << std::endl;

    // ------

    // weak2 points to new Foo
    if(std::shared_ptr<Foo> tmp = weak2.lock()){
        std::cout << "use count: " << ptr1.use_count() << std::endl;
        std::cout << "weak 2: ";
        tmp->show();
    }
    else
        std::cout << "weak2 is expired\n";

    return 0;
}

/**
 Application:
 - Cache
 - Team one-to-many members
 */
