/**
 * DEFINITION
 */

#include <iostream>
#include <cassert>
#include "Vector.hpp"

using std::cout, std::endl;


namespace mymath::myvector {
    int FOOBAR = 123;
    // FOOBAR = 123; // нельзя присваивать вне функции или вне определения

    // Constructor
    Vector::Vector() : NamedObject("DefaulVectorName"), _size(0) {
        cout << "Default Vector Constructor Call" << endl;
    }
    Vector::Vector(const int size) : Vector() {
        cout << "Vector Constructor Call" << endl;
        // Actually, we have access to _theData
        this->setSize(size);
    }

    // Copy constructor
    Vector::Vector(const Vector & other){
        cout << "Copy Vector Constructor Call" << endl;
        copy(this, &other);
    }

    // Destructor
    Vector::~Vector() {
        cout << "Vector Destructor Call: \n\t name(" << name << "), \n\t coeffs address(" << &_coeffs << ")" << endl;
        delete[] this->_coeffs;
    }

    // namespace {
        void copy(Vector * const self, const Vector * const other){
            cout << "Copying: other(" << other->getName() << ") to self(" << self->getName() << ")";
            cout << "\n\t self(" << self << "), coeffs address(" << self->_coeffs << ")";
            cout << "\n\t other(" << other << "), coeffs address(" << other->_coeffs << ")" << endl;

            assert(self != other && "stopped copying the same object");

            if(self->_coeffs){
                delete[] self->_coeffs;
                self->_coeffs = nullptr;
            }

            self->_size = other->_size;
            self->_coeffs = new int[self->_size];
            for(int i = 0; i < self->_size; i++){
                self->_coeffs[i] = other->_coeffs[i];
            }
        }
    // }

    // Setter
    void Vector::setSize(const int size){
        if(_coeffs) {
            cout << "Caution, overriding! The Vector has already been defined before!" << endl;
            delete[] _coeffs;
        }
        this->_size = size;
        this->_coeffs = new int[size];
        for(int i = 0; i < _size; i++){
            _coeffs[i] = 0;
        }
    }

    // Setter
    void Vector::set(const int scalar, const int index){
        // Actually, we have access to _theData
        this->_coeffs[index] = scalar;
    }
    // Getters
    int Vector::get(const int index) const{
        return this->_coeffs[index];
    }
    void Vector::get(const int index, int * const ptr_scalar) const{
        *ptr_scalar = this->_coeffs[index];
    }

    // Override Virtual Function of NamedObject
    void Vector::sayHello() const {
        cout << "Hello! I am Vector: \n\t name(" << name << "), \n\t coeffs address(" << &_coeffs << ")" << endl;
    }


    // Copy Operator
    Vector & Vector::operator = (const Vector & other) {
        if(this != &other){
            copy(this, &other);
        }
        return *this;
    }

    // Binary Multiplication Operator
    Vector operator * (const Vector & v1, const Vector & v2){
        Vector temp; // Stack memory allocation
        temp.setName("Temporary Vector");
        copy(&temp, &v1); // v1 copy
        temp *= v2; // умножаем v2 прямо на temp
        // Return Value Optimization (RVO)
        return temp;
    }


    void Vector::printCoeffs(){
        cout << "Vector(" << name << "): ";
        for(int i = 0; i < _size; i++){
            cout << _coeffs[i] << " ";
        }
        cout << endl;
    }
}
