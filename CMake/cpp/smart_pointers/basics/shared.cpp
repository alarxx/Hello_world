
#include <iostream>
#include <memory>

#include "Foo.hpp"

int main(){
    std::shared_ptr<Foo> ptr = std::make_shared<Foo>();

    {
        std::shared_ptr<Foo> ptr_copy = ptr;
        if(ptr)
            std::cout << "ptr не пуст" << std::endl;
        std::cout << "use count (copy): " << ptr.use_count() << std::endl;
        std::cout << "use count: " << ptr_copy.use_count() << std::endl;
    }

    std::cout << "use count: " << ptr.use_count() << std::endl;

    return 0;
}
