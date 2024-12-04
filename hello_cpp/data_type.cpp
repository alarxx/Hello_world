#include <iostream>


using namespace std;

typedef double real_t;

struct vector_t {
    size_t size;
    real_t * coeffs;
};

struct Shape{
    int arr[10];
};

void quadrature(Shape sh){ // call-by-value creates copy of struct even the arrays
// void quadrature(const Shape & sh){ // so, use const or call-by-reference to prevent copying
    sh.arr[0] = 123;
    cout << sh.arr[0] << endl;
    return;
}

int main() {
    vector_t x;
    x.size = 10;
    x.coeffs = new real_t[x.size];
    vector_t y = x; // the same as x, without & creates copy of x
    y.size = 11;
    cout << "Sizes:" << endl;
    cout << x.size << endl;
    cout << y.size << endl;

    Shape shape = {};
    for (int i = 0; i < 10; ++i) {
        shape.arr[i] = i + 1; // Populate the array
    }
    quadrature(shape);
    cout << shape.arr[0] << endl;

    return 0;
}

