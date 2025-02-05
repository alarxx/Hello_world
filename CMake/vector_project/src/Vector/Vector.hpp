/**
 * DECLARATIONS
 */

#pragma once
#ifndef _VECTOR_H_
#define _VECTOR_H_

#include "cassert"
#include "../mymath/mymath.hpp"

using mymath::NamedObject;

// Named scope
namespace mymath::myvector {

	// Inheritance
	// Multiple inheritance exists
	// По умолчанию private наследование, то есть все поля становятся private
	// Если выбрать protected Named, то все public и protected поля станут private
	// public ничего не меняет
	class Vector : public NamedObject {
		private:
		protected:
			// Encapsulation
			// in Python: _protected, __private
			int _size;
			int * _coeffs = nullptr;
			// Создается ли копия _theData при call-by-value
			// default arguments
		public:
			// Constructor
			Vector();
			Vector(const int size);
			// Copy constructor
			Vector(const Vector & other);
			// Destructor
			~Vector();

			// any functions defined in class are `inline` functions
			inline int getSize() const {
				return _size;
			}
			void setSize(const int size);

			// Setter
			void set(const int scalar, const int index);
			// Getter
			int get(const int index) const; // doesn't change the state of object
			// Для const instance (Data data) разрешены только вызовы const функций, так как они не меняют состояние объекта

			/**
			* Call-by-referece, but it is non-intuitive that we gonna change the argument:

			void getData(int & data);

			* Better use pointer:
			*/
			void get(const int index, int * const ptr_scalar) const; // the pointer is const


			// Polymorphism
			void sayHello() const override;

			// переопределение операторов =, +=, + и может быть <<

			// Copy Operator
			// friend Data & operator = (const Data & other); ?????
			Vector& operator = (const Vector& other);

			// По идее не inline
			friend void copy(Vector * const self, const Vector * const other);

			// Index Operator
			int operator [] (const int index){
				assert(_size && "Vector size is 0!");
				return this->_coeffs[index];
			}

			// Unary Operators
			// Elementwise: v1 *= v2;
			Vector &  operator *= (const Vector & other){
				assert(_size == other._size && "Vector sizes must be the same!");
				for(int i = 0; i < _size; i++) {
					_coeffs[i] *= other._coeffs[i];
				}
				return *this;
			}

			// Binary Operator
			// friend - не является членом класса, но имеет доступ к private
			// Note: no "self" vector argument, therefore we use "friend" keyword
			friend Vector operator * (const Vector & v1, const Vector & v2);

			void printCoeffs();
	};

}

#endif
