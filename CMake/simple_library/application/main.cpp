#include "Data/Data.h"
#include <iostream>


int main(){
    Data data;

    data.setData(123);

    int whatis;
    whatis = data.getData(); // functional style
    // data.getData(&whatis);

    std::cout << "data: " << whatis << "\n";

    return EXIT_SUCCESS;
}
