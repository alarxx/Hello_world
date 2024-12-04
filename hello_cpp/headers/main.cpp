#include <iostream>
#include <string>
#include "utils.h"
// Include Guards
#include "utils.h"

// Bad practice, huge library
// using namespace std;

// we can use "using" inside function scope
using std::cout;
using std::endl;


int main(int argc, char* argv[], char* envp[]) {
    using std::string;

    string name = "World";
    cout << greet(name) << endl;
    cout << SQUARE(5) << endl;

    cout << "Argument Count: " << argc << "\n";
    for (int i = 0; i < argc; ++i) {
        cout << "Argument[" << i << "]: " << argv[i] << "\n";
    }



    // Environment Variables
    // Terminal Session
    // export lolkek=12345
    for (int i = 0; envp[i] != nullptr; ++i) {
        cout << envp[i] << "\n";
    }

    setenv("My_Key", "My_Value", 1); // Устанавливаем переменную окружения
    const char* value = std::getenv("My_Key");
    cout << value << endl;

    return 0;
}
