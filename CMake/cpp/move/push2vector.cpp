// https://www.geeksforgeeks.org/stdmove-in-utility-in-c-move-semantics-move-constructors-and-move-assignment-operators/

// C++ program to implement
// the above approach

// for std::string
#include <string>

// for std::cout
#include <iostream>

// for EXIT_SUCCESS macro
#include <cstdlib>

// for std::vector
#include <vector>

// for std::move()
#include <utility>

// Declaration
std::vector<std::string> createAndInsert();

// Driver code
int main()
{
    // Constructing an empty vector
    // of strings
    std::vector<std::string> vecString;

    // calling createAndInsert() and
    // initializing the local vecString
    // object
    vecString = createAndInsert();

    // Printing content of the vector
    for (const auto& s : vecString) {
        std::cout << s << '\n';
    }

    return EXIT_SUCCESS;
}

// Definition
std::vector<std::string> createAndInsert()
{
    std::cout << "--- createAndInsert ---" << std::endl;

    // constructing a vector of
    // strings with an size of
    // 3 elements
    std::vector<std::string> vec;
    vec.reserve(3);

    // constructing & initializing
    // a string with "Hello"
    std::string str("Hello");

    std::cout << "0: " << str << std::endl;

    // Inserting a copy of string
    // object
    vec.push_back(str); // lvalue
    std::cout << "1: " << str << std::endl;

    // Inserting a copy of an
    // temporary string object
    vec.push_back(str + str); // rvalue
    std::cout << "2: " << str << std::endl;

    // Again inserting a copy of
    // string object
    vec.push_back(std::move(str)); // it may not create a copy
    // now str is empty
    std::cout << "3: " << str << std::endl;

    std::cout << "------" << std::endl;

    // Finally, returning the local
    // vector
    return vec;
}

