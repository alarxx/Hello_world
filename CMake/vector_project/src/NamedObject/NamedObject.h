#pragma once
#ifndef _NAMED_OBJECT_H_
#define _NAMED_OBJECT_H_

#include <iostream>
#include <string>
#include "../mymath/mymath.h"

namespace mymath {
	class NamedObject {
		protected: // Access modifier
			std::string name;
		public:
			NamedObject(){
				std::cout << "Default Constructor of NamedObject (no name)" << this->name << std::endl;
			}
			NamedObject(const std::string & name) {
				this->name = name;
				std::cout << "Constructor of NamedObject: \n\t name(" << name << ")" << std::endl;
			}

			/** Зачем Virtual Destructor?
			* Проблема может быть, когда мы вызываем delete pointer с типом SuperClass-а

			SuperClass * ptr = new ChildClass;
			delete ptr; // ptr type is SuperClass

			*
			*/
			virtual ~NamedObject(){
				std::cout << "Destructor of NamedObject: \n\t name(" << name << ")" << std::endl;
			}


			/** Virtal Functions (Abstraction and Polymorphism)
			*
			* NamedObject can be Abstract Class if we include Pure Virtual Function

			virtual void sayHello() const = 0;

			*
			* We can't define pure virtual functions of Abstract Class outside of the class declaration

			void NamedObject::sayHello() const {
				std::cout << "Hello! I am NamedObject!" << std::endl;
			}

			*
			**/
			virtual void sayHello() const {
				std::cout << "Hello! I am NamedObject: \n\t name(" << name << ")" << std::endl;
			}


			// Setter
			void setName(const std::string name){
				std::cout << "Set name: " << name << std::endl;
				this->name = name;
			}

			// Getter
			std::string getName() const {
				return this->name;
			}
			void getName(std::string * const name) const {
				*name = this->name;
			}
	};
}

#endif

