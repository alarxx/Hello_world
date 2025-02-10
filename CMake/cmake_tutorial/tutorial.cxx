/*
 * A simple program that computes the square root of a number
 */

#include <cmath>
// #include <cstdlib> // atof
#include <iostream>
#include <string>

#include "TutorialConfig.h"
#include "MathFunctions.h"

using std::cout;
using std::endl;

void print_cxx_std();


int main(int argc, char* argv[]){
  print_cxx_std();

  cout << "Version: " << Tutorial_VERSION_MAJOR << "." << Tutorial_VERSION_MINOR << endl;
  cout << MY_CUSTOM_VARIABLE << endl;

  #ifdef USE_MYMATH
    cout << "USE_MYMATH" << endl;
  #else
    cout << "USE_MYMATH not defined" << endl;
  #endif

  if (argc < 2) {
    cout << "Usage: " << argv[0] << " <number>" << endl;
    cout << "Pass a number, please!" << endl;
    return 1;
  }

  // convert string to double
  // const double inputValue = atof(argv[1]);
  const double inputValue = std::stod(argv[1]); // C++11 feature

  // calculate square root
  // const double outputValue = sqrt(inputValue);
  const double outputValue = mathfunctions::sqrt(inputValue);
  cout << "The square root of " << inputValue << " is " << outputValue << endl;

  return 0;
}


void print_cxx_std(){
// https://stackoverflow.com/questions/2324658/how-to-determine-the-version-of-the-c-standard-used-by-the-compiler
  if (__cplusplus == 202101L) std::cout << "C++23";
  else if (__cplusplus == 202002L) std::cout << "C++20";
  else if (__cplusplus == 201703L) std::cout << "C++17";
  else if (__cplusplus == 201402L) std::cout << "C++14";
  else if (__cplusplus == 201103L) std::cout << "C++11";
  else if (__cplusplus == 199711L) std::cout << "C++98";
  else std::cout << "pre-standard C++." << __cplusplus;
  std::cout << "\n";
}
