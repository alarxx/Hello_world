#include <iostream>
#include <memory>

void takeOwnership(std::unique_ptr<int> ptr){
	std::cout << "function owns: " << *ptr << std::endl;
}

int main(){
	std::unique_ptr<int> uptr = std::make_unique<int>(100);
    std::cout << "uptr: " << *uptr << std::endl;
	takeOwnership(std::move(uptr));
	if(!uptr){
        // Core dump, because uptr doesn't hold anything anymore
		std::cout << "uptr: " << *uptr << std::endl;
	}
}
