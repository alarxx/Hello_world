#include <iostream>

#include "Data/Data.h"

int main(){
    Data data;
    data.setData(123);
    int whatis = data.getData(); // functional style
    // data.getData(&whatis);
    std::cout << "data: " << whatis << "\n";

    return 0;
}
