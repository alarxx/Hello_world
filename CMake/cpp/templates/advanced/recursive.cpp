#include <iostream>
using std::cout, std::endl;

// --- Compile-time Factorial ---
template <int N>
class Factorial {
public:
    static constexpr int number = N * Factorial<N - 1>::number;
};

template <>
class Factorial<1>{
public:
    static constexpr int number = 1;
};

template <typename T> constexpr T factorial(const T N){
    T tmp = N;
    for(int i=2; i<N; i++){
        tmp *= i;
    }
    return tmp;
}
// ------

// --- Compile-time Log ---
template <int A, int B>
class Max {
public:
    static constexpr int value = A > B ? A : B;
};

template <int N, int BASE>
class Log {
public:
    static_assert(N > 0, "Number must be greater than zero");
    static_assert(BASE > 0, "Base must be greater than zero");

    static constexpr int TMP = Max<1, N / BASE>::value;
    static constexpr int value = 1 + Log<TMP, BASE>::value;
};
template <int BASE>
class Log<1, BASE> {
public:
    static constexpr int value = 0;
};
// ------

int main(){
    // constexpr int number = Factorial<5>::number;
    constexpr int number = factorial<int>(5);
    cout << number << endl;

    constexpr int log2_20 = Log<20, 2>::value;
    cout << log2_20 << endl;
}
