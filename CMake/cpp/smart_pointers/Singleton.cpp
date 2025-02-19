
// https://refactoring.guru/design-patterns/singleton/cpp/example

#include <iostream>

// --- Singleton ---

// Странный синтаксис. `= default`:
// `default` выглядит чище и оптимизирует как-то, потому что не имеет какого-то overhead-а.
class Singleton {
protected:
    int _value;
    static Singleton * _instance;
    // Singleton() = default;
    // Singleton(){} // Equivalent, but has some overhead?
    Singleton(int value) : _value(value){
        std::cout << "Singleton Constructor!" << std::endl;
    }
public:
	static Singleton * getInstance(int value = 0);
	// Not clonable
	Singleton(const Singleton &) = delete;
    // Copy Operator, Not assignable
	Singleton & operator = (const Singleton&) = delete;

    int value(){ return _value; }
};

Singleton * Singleton::_instance = nullptr;

Singleton * Singleton::getInstance(int value){
    if(_instance == nullptr){
       _instance = new Singleton(value);
    }
    return _instance;
}

// ------

// --- Singleton killer ---

class A {
public:
	static int variable;
};
int A::variable = 0;

// ------

int main(){
    // --- Singleton ---
    // Пример, как запрещать копирование
    Singleton * a = Singleton::getInstance(123);
    a = Singleton::getInstance();
    a = Singleton::getInstance();

    std::cout << "Singletong->value(): " << a->value() << std::endl;
    // Singleton b = *a; // error: use of deleted function ‘Singleton::Singleton(const Singleton&)’
    // Singleton c; // error: ‘constexpr Singleton::Singleton()’ is private within this context

    // --- Singleton killer ---
    // Но зачем так выделываться, если можно использовать
    std::cout << A::variable << std::endl;
    // Единственное, что мы можем потом где-то переназначить A::variable =

    return 0;
}
