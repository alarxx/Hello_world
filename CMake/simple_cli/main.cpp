#include <iostream>
#include <string>

int main(int argc, char * argv[]){
    std::string name;
    for(int i = 1; i < argc; i++){
        name += argv[i];
    }

    // trim
    name.erase(0, name.find_first_not_of(" \t\n\r\f\v"));
    name.erase(name.find_last_not_of(" \t\n\r\f\v") + 1);

    std::cout << name << std::endl;

    return EXIT_SUCCESS;
}
