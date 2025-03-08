
#include <iostream>
#include <cstdlib> // rand, srand
#include <ctime> // time


#include "iterator/iterator.hpp"
#include "iterator/constant_iterator.hpp"


template <typename T, int N = 0>
class Integers {
private:
    T _data[N];
public:
    using iterator = ::iterator<T>; // iterator variable shadowing, so we use :: - global namespace.
    using constant_iterator = ::constant_iterator<T>;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using constant_reverse_iterator = std::reverse_iterator<constant_iterator>;

    // _data[0] = *(_data + 0)
    // &(*(_data + 0)) = _data
    iterator begin(){ return iterator(_data); }
    iterator end(){ return iterator(&_data[N]); }

    constant_iterator cbegin() const { return constant_iterator(&_data[0]); }
    constant_iterator cend() const { return constant_iterator(&_data[N]); }

    reverse_iterator rbegin(){ return reverse_iterator(end()); }
    reverse_iterator rend(){ return reverse_iterator(begin()); }

    constant_reverse_iterator crbegin() const {
        return constant_reverse_iterator(cend()); // &_data[N - 1]
    }
    constant_reverse_iterator crend() const {
        return constant_reverse_iterator(cbegin()); // &_data[-1]
    }
};


int main(){
    Integers<int, 10> arr;

    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    time_t seconds = ts.tv_sec; // seconds
    long nanoseconds = ts.tv_nsec; // только nano часть
    long t = seconds * 1000000000l + nanoseconds; // total_nanoseconds

    // time_t t = time(NULL); // seconds

    srand(t);

    int i = 0;
    // for(auto it: arr) it = 10; // copy, don't actually changing the value
    for(auto & it: arr) {
        // it = rand() % 100;
        it = i++;
        // https://en.cppreference.com/w/cpp/numeric/random/rand
    }

    std::cout << typeid(decltype(arr)::iterator).name() << std::endl;

    // for(auto it: arr){
    // for(Integers<int, 10>::iterator it = arr.begin(); it != arr.end(); it++){
    // for(auto it = arr.begin(); it != arr.end(); it++){
    for(auto it = arr.rbegin(); it != arr.rend(); it++){
    // for(auto it = arr.crbegin(); it != arr.crend(); it++){
        it = 20;
        std::cout << *it << std::endl;
    }

}
