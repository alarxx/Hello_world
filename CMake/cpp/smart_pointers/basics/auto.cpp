// https://stackoverflow.com/questions/3697686/why-is-auto-ptr-being-deprecated

#include <iostream>
#include <memory>

#include "Foo.hpp"

int main() {
    std::auto_ptr<Foo> p1(new Foo); // `auto_ptr` создаёт объект
    std::auto_ptr<Foo> p2 = p1;       // p1 теперь пуст (владение передано)

    // ЭТО ПОЧЕМУ-ТО ДАЖЕ НЕ ЗАПУСКАЕТСЯ
    p1.show();
    p2.show();

    if (!p1)
        std::cout << "p1 теперь пуст\n";

    return 0; // `p2` удаляет объект
}

// Проблема с auto ptr в том, что ведет себя как unique_ptr, но при копировании передает владение, и непонятно
