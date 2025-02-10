#include "mysqrt.h"

#include <iostream>
#include <cmath>

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
      #if defined(HAVE_LOG) && defined(HAVE_EXP)
        double result = std::exp(0.5 * std::log(x));
        std::cout << "Computing sqrt of " << x << " to be " << result << " using log and exp" << std::endl;
        std::cout << "mysqrt HAVE_LOG: " << HAVE_LOG << std::endl;
        std::cout << "mysqrt HAVE_EXP: " << HAVE_EXP << std::endl;
      // elif defined(x)
      #else
        // initial guess
        double result = x;

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
        // Do ten iterations:
        for (int i = 0; i < 10; ++i) {
          if (result <= 0) {
            result = 0.1;
          }
          double delta = x - (result * result);
          result = result + 0.5 * delta / result;
          std::cout << "Computing sqrt of " << x << " to be " << result << std::endl;
        }
      #endif
      return result;
    }
  }
}
