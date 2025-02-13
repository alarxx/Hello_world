#include "MathFunctions.h"

#include <iostream>
#include <cmath>

#ifdef USE_MYMATH
  #include "mysqrt.h"
#endif

void print_cxx_std();

namespace mathfunctions
{
  double sqrt(double x)
  {
    print_cxx_std();

    #ifdef USE_MYMATH
// Это происходит на стадии препроцессинга, то есть
// в конечной версии будет один из двух вариантов
      return detail::mysqrt(x);
    #else
      return std::sqrt(x);
    #endif
  }
}

/*
Somehow on linux we can use this function with only one definition in tutorial.cxx
*/
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