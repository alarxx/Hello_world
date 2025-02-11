#include "mysqrt.h"

#include <iostream>
#include <cmath>

#include "Table.h"

namespace mathfunctions {
  namespace detail {
    // a hack square root calculation using simple operations
    double mysqrt(double x)
    {
      if (x <= 0) {
        return 0;
      }

      // Preprocessor directives
      // ifdef, ifndef
      #if USE_EXP_LOG != 1
        std::cout << "USE_EXP_LOG != 1: " << USE_EXP_LOG << std::endl;
      #elif USE_EXP_LOG == 1
        std::cout << "USE_EXP_LOG = 1: " << USE_EXP_LOG << std::endl;
      #endif

      #if defined(HAVE_LOG) && defined(HAVE_EXP)
      // variables defined только если USE_EXP_LOG == 1, check CMakeLists.txt
        double result = std::exp(0.5 * std::log(x));
        std::cout << "Computing sqrt of " << x << " to be " << result << " using exp and log" << std::endl;
        std::cout << "mysqrt HAVE_LOG: " << HAVE_LOG << std::endl;
        std::cout << "mysqrt HAVE_EXP: " << HAVE_EXP << std::endl;
      #else
        // initial guess
        // use the table to help find an initial value
        double result = x;
        if (x >= 1 && x < 10) {
          std::cout << "Use the table to help find an initial value " << std::endl;
          result = sqrtTable[static_cast<int>(x)];
          // Обычное округление числа.
          // По идее должен вернуть число ближайший к square root.
        }

        /* Newton's Root Finding Method
        * sqrt(n) = x
        * x=?
        * n = x^2 and x^2 - n = 0
        * f(x) = x^2 - n
        * f(x) = 0 - root
        * We need our initial guess and then iterativelly calculate:
        * x* = x - f(x) / f'(x)
        * f'(x) = 2*x
        */
        for (int i = 0; i < 10; ++i) {
        // Do ten iterations:
          if (result <= 0) {
            result = 0.1;
          }
          double delta = x - (result * result);
          result = result + 0.5 * delta / result;
          std::cout << "Computing sqrt of " << x << " to be " << result << " using Newton's method" << std::endl;
        }
      #endif
      return result;
    }
  }
}
