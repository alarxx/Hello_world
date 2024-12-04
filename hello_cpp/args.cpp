#include <iostream>
#include <string>

using std::cout;
using std::endl;
using std::string;


int main(int argc, char* argv[], char* envp[]) {

    // argc
    cout << "Argument Count: " << argc << "\n";

    // argv
    for (int i = 0; i < argc; ++i) {
        cout << "Argument[" << i << "]: " << argv[i] << "\n";
    }


    // envp - environment variables
    // Terminal Session: $export lolkek=12345
    for (int i = 0; envp[i] != nullptr; ++i) {
        cout << envp[i] << "\n";
    }


    // .ENV
    setenv("My_Key", "My_Value", 1); // Устанавливаем переменную окружения
    const char* value = std::getenv("My_Key");
    cout << value << endl;


    return 0;
}

