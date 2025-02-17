// https://www.w3schools.com/cpp/cpp_files.asp

#include <iostream>
#include <fstream>
#include <string> // string, getline

int main(){
    // out file stream
    std::ofstream MyFile("file.txt");

    MyFile << "Files are tricky! \n";

    MyFile.close();

    std::string myText;

    std::ifstream MyReadFile("file.txt");

    while(getline(MyReadFile, myText)){
        std::cout << myText;
    }
    std::cout << std::endl;

    MyReadFile.close();
}
