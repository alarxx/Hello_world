#include <iostream>
#include <memory>

#include "Foo.hpp"

int main(){
    std::unique_ptr<Foo> ptr = std::make_unique<Foo>();
    ptr->show();
    // std::unique_ptr<Foo> copy = ptr;
    // error: use of deleted function
    // то есть они запретили копирование

    // Пример, как запрещать копирование в Singleton.cpp

    std::unique_ptr<Foo> ptr_moved = std::move(ptr);

    if(!ptr){
        std::cout << "ptr теперь пуст" << std::endl;
    }

    std::cout << "From ptr_moved: ";
    ptr_moved->show();

    return 0;
}
