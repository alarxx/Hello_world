#include <iostream>

using std::cout;
using std::endl;

typedef unsigned int uint;

// function name scope
inline double sqrt(const double & x);

// double fun(double & var){ // call-by-reference
double fun(double var){ // call-by-value
    cout << "calculating fun" << endl;
    var = sqrt(var); // Won't change
    return var;
}

double fun(double var, int a, int b = 123){ // default argument
    cout << "default args fun" << endl;
    return fun(var);
}


// Нужно ли в двух местах писать inline?
inline double sqrt(const double & x){
    return x * x;
}

// Recursion example
uint fibonacci(const int n);


// static variable in function
void f ( const double x, long & cnt ) {
    static long counter = 0; // allocated and initialised
    // once per program
    cnt = ++counter;
}


int main(){

    // ++i - сначала ++, потом возврат
    // i++ - сначала возврат, потом ++
    int i = 0;
    long cnt = 0;
    for ( double x = 0; x <= 200; x += 1 ){
        cout << i++ << endl;
        f( x, cnt );
    }
    cout << cnt << endl; // print number of func. calls


    double var = 10;
    double retval;
    if(var >= 10 && (retval=fun(var, 0))){
        cout << var << endl;
        cout << retval << endl;
    }


    uint fibo = fibonacci(5);


    return 0;
}


// Recursion
// Можно declare-рировать после main, потому что код компилируется
uint fibonacci(const int n){
    cout << "Fibonacci " << n << endl;
    uint retval;
    if(n<=1){
        retval = 1;
    }
    else {
        uint fibo1 = fibonacci(n-1);
        uint fibo2 = fibonacci(n-2);
        retval = fibo1 + fibo2;
    }
    return retval;
}


double power (const double x, const unsigned int n){
    switch(n){
        case 0: return 1;
        case 1: return x;
        case 2: return square( x );
        default: {
            double f = x;
            for ( int i = 0; i < n; i++ )
                f *= x;
            return f;
        }
    }
}
