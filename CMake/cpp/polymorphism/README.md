# Polymorphism

References;
- https://www.w3schools.com/cpp/cpp_polymorphism.asp
- https://www.geeksforgeeks.org/cpp-polymorphism/
- https://cplusplus.com/doc/tutorial/polymorphism/

---

- **Compile-time Polymorphisim**
	- Function Overloading
	- Operator Overloading -> [[C C++ Programming Language#Operator Overloading|C C++ Operator Overloading]]
	- [[C++ Programming Language - template|templates]] относятся к compile-time polymorphism.
- **Runtime Polymorphism**
	- Virtual Functions

----

#### Function Overloading

Самый простой полиморфизм на уровне функций, когда есть множество функций с одинаковым именем, но разными аргументами.

 In C you can't create functions with the same name.
 But you can in C++ (Polymorphism):
```cpp
#include <stdio.h> // There is still C headers in C++, or <cstdio>

int     fun(int     x){ return x; }
double  fun(double  x){ return x; }
float   fun(float   x){ return x; }

int main(){
    printf("int:    %d \n", fun(1   ));
    printf("double: %f \n", fun(1.0 ));
    printf("float:  %f \n", fun(1.0f));
}
```

---

#### Inheritance

```cpp
// https://www.w3schools.com/cpp/cpp_polymorphism.asp

#include <iostream>
#include <string>

// Base class
class Animal {
public:
    void makeSound(){
        std::cout << "Animal sound!" << std::endl;
    }
...
};

// Multiple inheritance exists
// https://www.geeksforgeeks.org/cpp-inheritance-access/
// По умолчанию private наследование, то есть все поля становятся private
// Если выбрать protected, то все public и protected поля станут protected
// public ничего не меняет
class Dog : public Animal {
public:
	// Definition of the member function of the base class
    void makeSound() {
        std::cout << name << ": Bark bark!" << std::endl;
    }
	void bark(){
        std::cout << name << ": Bark bark!" << std::endl;
	}
...
};
```

```cpp
Animal animal("AnimalName");
animal.makeSound(); // > AnimalName: Animal sound!

Dog dog("DogName");
dog.makeSound(); // > DogName: Bark bark!
```

Да, мы можем переопределять функции и без использования `virtual`. Проблемы начинаются, если мы хотим использовать объект или указатель с типом базового класса, например:
```cpp
// Early binding, обрезка до базового класса
Animal dog = Dog("DogName");
dog.makeSound(); // > DogName: Animal sound!

Animal * dogptr = new Dog("DogNameP");
dogptr->makeSound(); // > DogNameP: Animal sound!
```

Вызвались методы базового класса!

---

Так же стоит сказать, что при обрезке до базового класса и при использовании указателя с типом базового класса, мы не можем получить методы производных классов:
```cpp
// Early binding, обрезка до базового класса
Animal dog = Dog("DogName");
dog.bark();
> error: ‘class Animal’ has no member named ‘bark’

Animal * dogptr = new Dog("DogNameP");
dogptr->bark();
> error: ‘class Animal’ has no member named ‘bark’
```


---

## `virtual`

- `virtual` functions
- `override`
- pure `virtual` functions and abstract classes (implementation must be provided)
- casting to and from base class, also [[C Programming Language - Type Casting]]

**virtual работает только для указателей и ссылок?** Да.
- Early binding - function is selected in compile-time
- Late binding (dynamic dispatch) - function is selected in runtime

**Можно ли создать instance класса с `virtual` функцией?**
Обычная `virtual` function всегда идет с имплементацией.
Pure `virtual` function идет без имплементации и делает класс абстрактным. Мы не можем создать instance абстрактного класса.

---

#### Virtual functions

Виртуальные функции обеспечивают полиморфизм, по типу `@Override` в Java, только если бы не виртуальные функции, вызов зависел бы от типа указателя на объект, а не от выделенного объекта:
```cpp
#include <iostream>

// Base class
class Animal {
public:
	virtual void makeSound() { // Виртуальная функция
        std::cout << name << ": Animal sound!" << std::endl;
	}
};

// Derivative class
class Dog : public Animal {
public:
    void makeSound() override { // Переопределяем виртуальную функцию
        std::cout << name << ": Bark bark!" << std::endl;
    }
	void bark(){
        std::cout << name << ": Bark bark!" << std::endl;
	}
};
```

```cpp
Animal dog = Dog("DogName");
// Early binding - function is selected at compile time.
dog.makeSound(); // "Animal sound!", even with virtual function
// dog.bark();
// error: ‘class Animal’ has no member named ‘bark’

Animal * dogptr = new Dog("DogNameP"); // pointer
Animal & dogref = *dogptr; // reference
// dogptr->introduce();
dogref.introduce();
// Late binding (dynamic dispatch) - function is selected in runtime.
// dogptr->makeSound(); // "Bark bark!"
dogref.makeSound(); // "Bark bark!"
// dogptr->bark();
// error: ‘class Animal’ has no member named ‘bark’

delete dogptr;
```

```sh
> DogName: Animal sound!
> DogNameP: Bark bark!
```

---

#### Pure virtual functions

**Pure virtual functions** - абстрактные функции, которые обязаны реализовать наследники, и если в классе есть одна чисто виртуальная функция, то класс является абстранктным:
```cpp
// Абстрактный класс class
class Animal {
	public:
		virtual void speak() = 0; // Чисто виртуальная функция
};
```

Нельзя создать экземпляр абстрактного класса:
```cpp
Animal animal;

> error: cannot declare variable ‘animal’ to be of abstract type ‘Animal’
> note:   because the following virtual functions are pure within ‘Animal’
```

----

#### Smart pointers

```cpp
std::unique_ptr<Animal> dogptr = std::make_unique<Dog>("DogNameP");
dogptr->introduce();
// Late binding (dynamic dispatch) - function is selected in runtime.
dogptr->makeSound(); // "Bark bark!"
// dogptr->bark();
// error: ‘class Animal’ has no member named ‘bark’
```

---

#### Casting

Есть C-style casting, как здесь -> [[C Programming Language - Type Casting]].

**References:**
- https://stackoverflow.com/questions/28002/regular-cast-vs-static-cast-vs-dynamic-cast
- https://stackoverflow.com/questions/332030/when-should-static-cast-dynamic-cast-const-cast-and-reinterpret-cast-be-used

**C++ type casting:**
- C-style type casting (static + dynamic + reinterpret_cast + const_cast)
- `static_cast`, runtime cast errors is undefined behavior
- `dynamic_cast`, additionally checking in runtime and can return `nullptr` or throw `std::bad_cast`
- `reinterpret_cast`, straightforward
- `const_cast`, взлом const, кстати еще можно pointer-ами взломать

**C-style casting works:**
```cpp
Animal * dogptr = new Dog("DogNameP");
void * void_dogptr = (void *) dogptr;
((Animal *) void_dogptr)->makeSound(); // > "Bark bark!"
// Но, Animal не видит bark()
// Поэтому мы приводим к (Dog *):
((Dog *) void_dogptr)->bark(); // > "Bark bark!"

// This works too!
((Cat *) dogptr)->makeSound(); // > "Bark bark!"
```

**`static_cast` тоже сработает и выдаст то же самое:**
```cpp
(static_cast<Cat *>(casted_dogptr))->makeSound(); // > "Bark bark!"
```
Actually, здесь Undefined Behavior.

And this is insane:
```cpp
(static_cast<Cat *>(dogptr))->meow(); // > "Meow meow!"
```

Мы здесь точно знаем, что Dog это Animal и т.д., если бы мы не были уверены, i.g. `(Cat *) dogptr`, то лучше использовать dynamic_cast.

**Как проверить, что типы совпадают?**
```cpp
// --- Pointer ---
Animal * dogptr = new Dog("DogNameP");
// `static_cast` нам бы разрешил:
// Cat * casted_dogptr = dynamic_cast<Cat *>(dogptr);
Dog * casted_dogptr = dynamic_cast<Dog *>(dogptr);
if(!casted_dogptr){
	std::cout << "cast failed!" << std::endl;
	return 1;
}
casted_dogptr->makeSound();

delete dogptr;
dogptr = nullptr;
// ------

// --- Reference ---
// Dog * catptr = (Dog *) new Cat("CatNameP"); // powerful
Animal * catptr = new Cat("CatNameP");
try{
	Dog & catref = dynamic_cast<Dog &>(*catptr); // throws std::bad_cast
	catref.makeSound();
}
// https://en.cppreference.com/w/cpp/types/bad_cast
catch (const std::bad_cast& e){
	std::cout << "Error handled: e.what(): " << e.what() << '\n';
}
delete catptr;
catptr = nullptr;
// ------
```


