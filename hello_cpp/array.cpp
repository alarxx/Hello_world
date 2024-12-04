#include <iostream>
#include <array>
// #include <cstring> // strlen()
#include <string>

using std::cout;
using std::endl;
using std::begin;
using std::end;
using std::array;

// void print_array(const int arr[7]){ // but you should specify the length
void print_array(const int * arr){
    for(int i=-1; i<10; i++){ // no segmentation fault i.e. no errors somehow
        cout << "arr[" << i << "]: " << arr[i] << endl;
    }
}

// Multidimensional Arrays
void mulvec(const double M[3][3], const double x[3], double y[3]){
    for ( int i = 0; i < 3; i++ ){
        y[i] = 0.0;
        for ( int j = 0; j < 3; j++ )
            y[i] += M[i][j] * x[j];
    }
}

// Example of Memory Leak - Lost Memory
// void f (){
//     double * v = new double[ 1000 ];
//     ... // no delete[] v
// }

int main(){
    int l = 10;

    int arr[l]; // lenght is constant
    l = 15;

    int arr2[] = {2, 5, 8, 11, 14, 17, 20};

    int * p = arr2;
    // print_array(p);


    /* Dynamic Memory */

    double * ptrDouble = new double;
    *ptrDouble = 9.5;
    int * arr3 = new int[10];

    cout << "\nInitialize array\n" << endl;
    for(int i=0/*-1*/; i<11; i++){ //
        // Lol мы хер пойми какой-то адрес перезаписали,
        // но вроде мы имеем доступ только к памяти нашей программы, а иначе выйдет ошибка
        arr3[i] = i;
        cout << arr3[i] << endl;
    }

    cout << "ptrDouble:" << *ptrDouble << endl;

    delete ptrDouble;
    // ptrDouble = NULL; // We should do this
    delete[] arr3;
    // arr3 = NULL; // We should do this

    // И нет способа понять, что указатели больше не действительны
    cout << "\nAfter delete[] array\n" << endl;
    for(int i=0/*-1*/; i<11; i++){ //
        cout << arr3[i] << endl;
    }

    cout << "ptrDouble: " << *ptrDouble << endl;

    std::string str = "Hal0lo ";
    cout << int(' ') << endl;
    cout << '0' << endl;
    cout << "\\0: " << '\0' << ", int('\\0'): " << int('\0') << endl;

    cout << str.length() << endl;

/*
    // How to know length of the array?

    // 1)
    cout << sizeof(arr2) / sizeof(arr2[0]) << endl;

    // 2)
    cout << sizeof(arr2) << endl;
    cout << arr2 << endl;
    cout << *arr2 << endl;
    int len = *(&arr2 + 1) - arr2;
    cout << "array_length:// 1)
    cout << sizeof(arr2) / sizeof(arr2[0]) << endl; " << len << endl;

    // 3) Хз как это работает
    cout << "Auto" << endl;
    for(auto i: arr2){
        cout << i << endl;
    }

    // 4)
    cout << end(arr2) - begin(arr2) << endl;

    // 5)
    array<int, 5> arr3 = {1, 2, 3, 4, 5};
    cout << "size(): " << arr3.size() << endl;
*/

/*
    double castd = 123.567d;
    cout << (int) castd << endl; // C
    cout << int(castd) << endl; // C++
    cout << static_cast<int>(castd) << endl; // C++
*/



}
