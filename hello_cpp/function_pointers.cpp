#include <iostream>

using std::cout;
using std::endl;

/*
    function is stored somewhere in the memory and therefore also has an address.

    Hence, a pointer can be acquired to a function

    type (* name)(args)
*/

// Simpson rule ?
double simpson_quad (const double a, const double b, double ( * func ) ( const double ) ){
    return (b-a) / 6.0 * ( func(a) + 4 * func( (a+b) / 2.0 ) + func(b) );
}

double f1 ( const double x ) {
    return x*x;
}

double f2 ( const double x ) {
    return x*x*x;
}

int main (){
    cout << simpson_quad( -1, 2, f1 ) << endl;
    cout << simpson_quad( -1, 2, f2 ) << endl;
}
