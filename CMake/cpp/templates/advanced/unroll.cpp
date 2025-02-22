#include <iostream>
using std::cout, std::endl;

template <int NUM_UNROLL, int STEP = 0>
class Unroll {
public:
    template <typename Op>
    static void run(Op op){
        op(STEP);
        Unroll<NUM_UNROLL, STEP + 1>::run(op);
    }
};

template <int NUM_UNROLL>
class Unroll<NUM_UNROLL, NUM_UNROLL> {
public:
    template <typename Op>
    static void run(Op op){
        op(NUM_UNROLL);
    }
};

int main(){
    auto lambda = [](int n){
        cout << n << endl;
    };
    Unroll<5>::run(lambda);
}
