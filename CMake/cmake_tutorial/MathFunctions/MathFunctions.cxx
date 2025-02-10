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
