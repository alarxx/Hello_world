#include <iostream>

template <typename ... TArgs>
class Tuple;

template <typename T>
class Tuple<T> { // Base Case
public:
    T value;
    Tuple(T value) : value(value) {}
};

template <typename T, typename ... TArgs>
class Tuple<T, TArgs...> {
public:
    T value;
    Tuple<TArgs...> tail;
    Tuple(T value, TArgs ... args) : value(value), tail(args...) {}
};

// Deduction Guide to Support Type Deduction
template <typename ... TArgs>
Tuple(TArgs ...) -> Tuple<TArgs...>;

int main(){
    // Tuple tuple(2, 2.0, 'a');
    Tuple tuple = {2, 2.0, 'a'};

    std::cout << tuple.value << std::endl;
    std::cout << tuple.tail.value << std::endl;
    std::cout << tuple.tail.tail.value << std::endl;
}
